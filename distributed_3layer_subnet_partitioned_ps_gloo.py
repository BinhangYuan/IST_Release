import time
from ist_utilis import *
from random import shuffle, seed
import torch.nn as nn
import torch.nn.functional as functional
import argparse
import numpy as np
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import google_speech_data_loader as speech_dataset
from torch.utils.data import DataLoader


class DNNGoogleSpeechBatchNorm3LayerModel(nn.Module):
    def __init__(self, args, feature_size=4096, model_size=4096, label_num=35):
        super(DNNGoogleSpeechBatchNorm3LayerModel, self).__init__()
        self.fc1 = nn.Linear(feature_size, model_size, False)
        self.bn1 = nn.BatchNorm1d(model_size, momentum=1.0, track_running_stats=False)
        self.fc2 = nn.Linear(model_size, model_size, False)
        self.bn2 = nn.BatchNorm1d(model_size, momentum=1.0, track_running_stats=False)
        self.fc3 = nn.Linear(model_size, label_num, False)
        self.bn3 = nn.BatchNorm1d(label_num, momentum=1.0, affine=False, track_running_stats=False)
        self.cluster_size = args.world_size

        self.fc1_weight_buffer = []
        self.fc1_weight_buffer_async_handle = []
        self.fc2_weight_buffer = []
        self.fc2_weight_buffer_async_handle = []
        self.fc3_weight_buffer = []
        self.fc3_weight_buffer_async_handle = []
        self.bn1_weight_buffer = []
        self.bn1_weight_buffer_async_handle = []
        self.bn1_bias_buffer = []
        self.bn1_bias_buffer_async_handle = []
        self.bn2_weight_buffer = []
        self.bn2_weight_buffer_async_handle = []
        self.bn2_bias_buffer = []
        self.bn2_bias_buffer_async_handle = []

        for _ in range(self.cluster_size):
            self.fc1_weight_buffer.append(torch.zeros([model_size//self.cluster_size, feature_size]))
            self.fc1_weight_buffer_async_handle.append(None)
            # Assume the PS is partition the weight matrix by dim 1 (its' input dim)
            self.fc2_weight_buffer.append(torch.zeros([model_size, model_size//self.cluster_size]))
            self.fc2_weight_buffer_async_handle.append(None)
            self.fc3_weight_buffer.append(torch.zeros([label_num, model_size//self.cluster_size]))
            self.fc3_weight_buffer_async_handle.append(None)
            self.bn1_weight_buffer.append(torch.zeros(model_size//self.cluster_size))
            self.bn1_weight_buffer_async_handle.append(None)
            self.bn1_bias_buffer.append(torch.zeros(model_size//self.cluster_size))
            self.bn1_bias_buffer_async_handle.append(None)
            self.bn2_weight_buffer.append(torch.zeros(model_size//self.cluster_size))
            self.bn2_weight_buffer_async_handle.append(None)
            self.bn2_bias_buffer.append(torch.zeros(model_size//self.cluster_size))
            self.bn2_bias_buffer_async_handle.append(None)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = functional.relu(x, inplace=True)
        x = self.fc2(x)
        x = self.bn2(x)
        x = functional.relu(x, inplace=True)
        x = self.fc3(x)
        x = self.bn3(x)
        x = functional.log_softmax(x, dim=1)
        return x

    def update_model_from_buffer(self):
        print('update_model_from_buffer')
        self.fc1.weight.data = torch.cat(self.fc1_weight_buffer, dim=0)
        self.fc2.weight.data = torch.cat(self.fc2_weight_buffer, dim=1)
        self.fc3.weight.data = torch.cat(self.fc3_weight_buffer, dim=1)
        self.bn1.weight.data = torch.cat(self.bn1_weight_buffer, dim=0)
        self.bn1.bias.data = torch.cat(self.bn1_bias_buffer, dim=0)
        self.bn2.weight.data = torch.cat(self.bn2_weight_buffer, dim=0)
        self.bn2.bias.data = torch.cat(self.bn2_bias_buffer, dim=0)

    def split_to_chunks(self):
        self.fc1_weight_buffer.clear()
        self.fc2_weight_buffer.clear()
        self.fc3_weight_buffer.clear()
        self.bn1_weight_buffer.clear()
        self.bn1_bias_buffer.clear()
        self.bn2_weight_buffer.clear()
        self.bn2_bias_buffer.clear()
        self.fc1_weight_buffer = [t.clone() for t in torch.chunk(self.fc1.weight.data, self.cluster_size, 0)]
        self.fc2_weight_buffer = [t.clone() for t in torch.chunk(self.fc2.weight.data, self.cluster_size, 1)]
        self.fc3_weight_buffer = [t.clone() for t in torch.chunk(self.fc3.weight.data, self.cluster_size, 1)]
        self.bn1_weight_buffer = [t.clone() for t in torch.chunk(self.bn1.weight.data, self.cluster_size, 0)]
        self.bn1_bias_buffer = [t.clone() for t in torch.chunk(self.bn1.bias.data, self.cluster_size, 0)]
        self.bn2_weight_buffer = [t.clone() for t in torch.chunk(self.bn2.weight.data, self.cluster_size, 0)]
        self.bn2_bias_buffer = [t.clone() for t in torch.chunk(self.bn2.bias.data, self.cluster_size, 0)]


class DNNGoogleSpeechBatchNorm3LayerPS:
    # here the model size means the size that stores in this parameter server
    def __init__(self, rank, partition_num=1, sample_size=4096, partitioned_model_size=4096, label_num=35):
        # super(DNNGoogleSpeechBatchNorm3LayerPS, self).__init__()
        self.rank = rank
        self.partition_num = partition_num
        self.partition_dim = partitioned_model_size // partition_num
        # Now each PS has the global view of the shuffle process so that it can get the information without sync.
        self.global_hidden_layer_index1 = []
        for _ in range(partition_num):
            self.global_hidden_layer_index1.append([i for i in range(partitioned_model_size)])
        self.global_hidden_layer_index2 = []
        for _ in range(partition_num):
            self.global_hidden_layer_index2.append([i for i in range(partitioned_model_size)])
        self.fc1 = nn.Linear(sample_size, partitioned_model_size, False)
        self.bn1 = nn.BatchNorm1d(partitioned_model_size, momentum=1.0, track_running_stats=False)
        self.fc2 = nn.Linear(partitioned_model_size, partitioned_model_size * partition_num, False)
        self.bn2 = nn.BatchNorm1d(partitioned_model_size, momentum=1.0, track_running_stats=False)
        self.fc3 = nn.Linear(partitioned_model_size, label_num, False)
        # The following is used for distributed training.
        assert partition_num != 1
        self.fc1_weight_partition_buffer_async_handle = []
        self.fc2_weight_partition_buffer_async_handle = []
        self.fc3_weight_partition_buffer_async_handle = []
        self.bn1_weight_partition_buffer_async_handle = []
        self.bn1_bias_partition_buffer_async_handle = []
        self.bn2_weight_partition_buffer_async_handle = []
        self.bn2_bias_partition_buffer_async_handle = []
        for _ in range(partition_num):
            self.fc1_weight_partition_buffer_async_handle.append(None)
            self.fc2_weight_partition_buffer_async_handle.append(None)
            self.fc3_weight_partition_buffer_async_handle.append(None)
            self.bn1_weight_partition_buffer_async_handle.append(None)
            self.bn1_bias_partition_buffer_async_handle.append(None)
            self.bn2_weight_partition_buffer_async_handle.append(None)
            self.bn2_bias_partition_buffer_async_handle.append(None)

        self.local_hidden_layer_index1_sampling = []
        self.local_hidden_layer_index2_sampling = []
        self.global_hidden_layer_index2_sampling = []
        self.fc1_weight_partition_buffer = []
        self.fc2_weight_partition_buffer = []
        self.fc3_weight_partition_buffer = []
        self.bn1_weight_partition_buffer = []
        self.bn1_bias_partition_buffer = []
        self.bn2_weight_partition_buffer = []
        self.bn2_bias_partition_buffer = []

    def partition_to_list(self):
        print("Repartition parameters!")
        for i in range(self.partition_num):
            shuffle(self.global_hidden_layer_index1[i])
            shuffle(self.global_hidden_layer_index2[i])
        self.local_hidden_layer_index1_sampling.clear()
        self.local_hidden_layer_index2_sampling.clear()
        self.global_hidden_layer_index2_sampling.clear()
        # generate the local partition for hidden dim 1
        for i in range(self.partition_num):
            current_local_indexes1 = torch.tensor(
                self.global_hidden_layer_index1[self.rank][i * self.partition_dim:(i + 1) * self.partition_dim])
            self.local_hidden_layer_index1_sampling.append(current_local_indexes1)
        # generate the local partition for hidden dim 2
        for i in range(self.partition_num):
            current_local_indexes2 = torch.tensor(
                self.global_hidden_layer_index2[self.rank][i * self.partition_dim:(i + 1) * self.partition_dim])
            self.local_hidden_layer_index2_sampling.append(current_local_indexes2)
        # generate the global hidden dim for hidden dim 2
        # (this is determined by the way we split the parameter weight in layer 2)
        for i in range(self.partition_num):  # for current PS's local partition
            current_global_hidden_index2 = []
            for j in range(self.partition_num):  # go through every partition in each PS
                current_global_hidden_index2.extend(
                    self.global_hidden_layer_index2[i][j*self.partition_dim:(j+1)*self.partition_dim])
            self.global_hidden_layer_index2_sampling.append(torch.tensor(current_global_hidden_index2))
        self.fc1_weight_partition_buffer.clear()
        self.fc2_weight_partition_buffer.clear()
        self.fc3_weight_partition_buffer.clear()
        self.bn1_weight_partition_buffer.clear()
        self.bn1_bias_partition_buffer.clear()
        self.bn2_weight_partition_buffer.clear()
        self.bn2_bias_partition_buffer.clear()
        self.fc1_weight_partition_buffer = partition_FC_layer_by_output_dim_0(
            self.fc1.weight, self.local_hidden_layer_index1_sampling)
        # This is an very important step
        self.fc2_weight_partition_buffer = partition_FC_layer_by_dim_01(
            self.fc2.weight, self.global_hidden_layer_index2_sampling, self.local_hidden_layer_index1_sampling)
        self.fc3_weight_partition_buffer = partition_FC_layer_by_input_dim_1(
            self.fc3.weight, self.local_hidden_layer_index2_sampling)
        self.bn1_weight_partition_buffer, self.bn1_bias_partition_buffer = partition_BN_layer(
            self.bn1.weight, self.bn1.bias, self.local_hidden_layer_index1_sampling)
        self.bn2_weight_partition_buffer, self.bn2_bias_partition_buffer = partition_BN_layer(
            self.bn2.weight, self.bn2.bias, self.local_hidden_layer_index2_sampling)

    def update_model_from_partition_buffer(self):
        # update the model based on the collected parameters.
        # Here we have to get around pytorch variable by use variable.data,
        # Since leaf variable disallowing in-place operation.
        update_tensor_by_update_lists_dim_0(self.fc1.weight.data, self.fc1_weight_partition_buffer,
                                            self.local_hidden_layer_index1_sampling)
        update_tensor_by_update_lists_dim_01(self.fc2.weight.data, self.fc2_weight_partition_buffer,
                                             self.global_hidden_layer_index2_sampling,
                                             self.local_hidden_layer_index1_sampling)
        update_tensor_by_update_lists_dim_1(self.fc3.weight.data, self.fc3_weight_partition_buffer,
                                            self.local_hidden_layer_index2_sampling)
        update_tensor_by_update_lists_dim_0(self.bn1.weight.data, self.bn1_weight_partition_buffer,
                                            self.local_hidden_layer_index1_sampling)
        update_tensor_by_update_lists_dim_0(self.bn1.bias.data, self.bn1_bias_partition_buffer,
                                            self.local_hidden_layer_index1_sampling)
        update_tensor_by_update_lists_dim_0(self.bn2.weight.data, self.bn2_weight_partition_buffer,
                                            self.local_hidden_layer_index2_sampling)
        update_tensor_by_update_lists_dim_0(self.bn2.bias.data, self.bn2_bias_partition_buffer,
                                            self.local_hidden_layer_index2_sampling)


def sync_communication(args, buffer_async_handle):
    for i in range(args.world_size):
        buffer_async_handle[i].wait()


def start_broadcast(args, model_prime_buffer, model_ps_data, model_worker_buffer_async_handle):
    for i in range(args.world_size):
        if i == args.rank:
            model_prime_buffer[i] = model_ps_data.clone()
        model_worker_buffer_async_handle[i] = dist.broadcast(tensor=model_prime_buffer[i], src=i, async_op=True)


def dispatch_model_for_validation(args, model_prime: DNNGoogleSpeechBatchNorm3LayerModel,
                                  model_ps: DNNGoogleSpeechBatchNorm3LayerPS):
    print("dispatch_model_to_prime_node")
    # collect the whole model from the distributed parameter servers
    start_broadcast(args, model_prime.fc1_weight_buffer, model_ps.fc1.weight.data,
                    model_prime.fc1_weight_buffer_async_handle)
    start_broadcast(args, model_prime.fc2_weight_buffer, model_ps.fc2.weight.data,
                    model_prime.fc2_weight_buffer_async_handle)
    start_broadcast(args, model_prime.fc3_weight_buffer, model_ps.fc3.weight.data,
                    model_prime.fc3_weight_buffer_async_handle)
    start_broadcast(args, model_prime.bn1_weight_buffer, model_ps.bn1.weight.data,
                    model_prime.bn1_weight_buffer_async_handle)
    start_broadcast(args, model_prime.bn1_bias_buffer, model_ps.bn1.bias.data,
                    model_prime.bn1_bias_buffer_async_handle)
    start_broadcast(args, model_prime.bn2_weight_buffer, model_ps.bn2.weight.data,
                    model_prime.bn2_weight_buffer_async_handle)
    start_broadcast(args, model_prime.bn2_bias_buffer, model_ps.bn2.bias.data,
                    model_prime.bn2_bias_buffer_async_handle)
    sync_communication(args, model_prime.bn1_weight_buffer_async_handle)
    sync_communication(args, model_prime.bn1_bias_buffer_async_handle)
    sync_communication(args, model_prime.bn2_weight_buffer_async_handle)
    sync_communication(args, model_prime.bn2_bias_buffer_async_handle)
    sync_communication(args, model_prime.fc3_weight_buffer_async_handle)
    sync_communication(args, model_prime.fc2_weight_buffer_async_handle)
    sync_communication(args, model_prime.fc1_weight_buffer_async_handle)


def start_scatter_ps2worker(args, worker_buffer, ps_partition_buffer, worker_buffer_async_handle):
    for i in range(args.world_size):
        if i == args.rank:
            worker_buffer_async_handle[i] = dist.scatter(tensor=worker_buffer[i], scatter_list=ps_partition_buffer,
                                                         src=i, async_op=True)
        else:
            worker_buffer_async_handle[i] = dist.scatter(tensor=worker_buffer[i], scatter_list=[], src=i, async_op=True)


def dispatch_model_to_workers(args, model_worker: DNNGoogleSpeechBatchNorm3LayerModel,
                              model_ps: DNNGoogleSpeechBatchNorm3LayerPS):
    start_scatter_ps2worker(args, model_worker.fc1_weight_buffer, model_ps.fc1_weight_partition_buffer,
                            model_worker.fc1_weight_buffer_async_handle)
    start_scatter_ps2worker(args, model_worker.fc2_weight_buffer, model_ps.fc2_weight_partition_buffer,
                            model_worker.fc2_weight_buffer_async_handle)
    start_scatter_ps2worker(args, model_worker.fc3_weight_buffer, model_ps.fc3_weight_partition_buffer,
                            model_worker.fc3_weight_buffer_async_handle)
    start_scatter_ps2worker(args, model_worker.bn1_weight_buffer, model_ps.bn1_weight_partition_buffer,
                            model_worker.bn1_weight_buffer_async_handle)
    start_scatter_ps2worker(args, model_worker.bn1_bias_buffer, model_ps.bn1_bias_partition_buffer,
                            model_worker.bn1_bias_buffer_async_handle)
    start_scatter_ps2worker(args, model_worker.bn2_weight_buffer, model_ps.bn2_weight_partition_buffer,
                            model_worker.bn2_weight_buffer_async_handle)
    start_scatter_ps2worker(args, model_worker.bn2_bias_buffer, model_ps.bn2_bias_partition_buffer,
                            model_worker.bn2_bias_buffer_async_handle)
    sync_communication(args, model_worker.bn1_weight_buffer_async_handle)
    sync_communication(args, model_worker.bn1_bias_buffer_async_handle)
    sync_communication(args, model_worker.bn2_weight_buffer_async_handle)
    sync_communication(args, model_worker.bn2_bias_buffer_async_handle)
    sync_communication(args, model_worker.fc3_weight_buffer_async_handle)
    sync_communication(args, model_worker.fc2_weight_buffer_async_handle)
    sync_communication(args, model_worker.fc1_weight_buffer_async_handle)


def start_scatter_worker2ps(args, worker_buffer, ps_partition_buffer, ps_buffer_async_handle):
    for i in range(args.world_size):
        if i != args.rank:
            ps_buffer_async_handle[i] = dist.scatter(tensor=ps_partition_buffer[i], scatter_list=[],
                                                     src=i, async_op=True)
        else:
            ps_buffer_async_handle[i] = dist.scatter(tensor=ps_partition_buffer[i], scatter_list=worker_buffer,
                                                     src=args.rank, async_op=True)


def dispatch_updated_model_to_ps(args, model_worker: DNNGoogleSpeechBatchNorm3LayerModel,
                                 model_ps: DNNGoogleSpeechBatchNorm3LayerPS):
    start_scatter_worker2ps(args, model_worker.fc1_weight_buffer, model_ps.fc1_weight_partition_buffer,
                            model_ps.fc1_weight_partition_buffer_async_handle)
    start_scatter_worker2ps(args, model_worker.fc2_weight_buffer, model_ps.fc2_weight_partition_buffer,
                            model_ps.fc2_weight_partition_buffer_async_handle)
    start_scatter_worker2ps(args, model_worker.fc3_weight_buffer, model_ps.fc3_weight_partition_buffer,
                            model_ps.fc3_weight_partition_buffer_async_handle)
    start_scatter_worker2ps(args, model_worker.bn1_weight_buffer, model_ps.bn1_weight_partition_buffer,
                            model_ps.bn1_weight_partition_buffer_async_handle)
    start_scatter_worker2ps(args, model_worker.bn1_bias_buffer, model_ps.bn1_bias_partition_buffer,
                            model_ps.bn1_bias_partition_buffer_async_handle)
    start_scatter_worker2ps(args, model_worker.bn2_weight_buffer, model_ps.bn2_weight_partition_buffer,
                            model_ps.bn2_weight_partition_buffer_async_handle)
    start_scatter_worker2ps(args, model_worker.bn2_bias_buffer, model_ps.bn2_bias_partition_buffer,
                            model_ps.bn2_bias_partition_buffer_async_handle)
    sync_communication(args, model_ps.bn1_weight_partition_buffer_async_handle)
    sync_communication(args, model_ps.bn1_bias_partition_buffer_async_handle)
    sync_communication(args, model_ps.bn2_weight_partition_buffer_async_handle)
    sync_communication(args, model_ps.bn2_bias_partition_buffer_async_handle)
    sync_communication(args, model_ps.fc3_weight_partition_buffer_async_handle)
    sync_communication(args, model_ps.fc2_weight_partition_buffer_async_handle)
    sync_communication(args, model_ps.fc1_weight_partition_buffer_async_handle)


def train(args, model_ps: DNNGoogleSpeechBatchNorm3LayerPS,
          model_worker: DNNGoogleSpeechBatchNorm3LayerModel,
          optimizer, train_loader, epoch, train_time_log):
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        if i < len(train_loader) // args.world_size:
            if i % args.repartition_iter == 0:
                model_ps.partition_to_list()
                dispatch_model_to_workers(args, model_worker, model_ps)
                model_worker.update_model_from_buffer()
            data, target = batch['wav'].float(), batch['label']
            optimizer.zero_grad()
            output = model_worker(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            train_correct = train_pred.eq(target.view_as(train_pred)).sum().item()
            if i % args.log_interval == 0:
                print('Train Epoch {} iter {} <Loss: {:.6f}, Accuracy: {:.2f}%>'.format(
                    epoch, i, loss.item(), 100. * train_correct / target.shape[0]))
            if (i + 1) % args.repartition_iter == 0 or i == len(train_loader) // args.world_size:
                model_worker.split_to_chunks()
                dispatch_updated_model_to_ps(args, model_worker, model_ps)
                model_ps.update_model_from_partition_buffer()
        else:
            break
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Node {}: Train Epoch {} total time {:3.2f}s'.format(args.rank, epoch, elapsed_time))
    if args.rank == 0:
        train_time_log[epoch-1] = elapsed_time


def sync_test_results(args, test_loss, test_correct, test_total):
    dist.reduce(tensor=test_loss, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(tensor=test_correct, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(tensor=test_total, dst=0, op=dist.ReduceOp.SUM)
    if args.rank == 0:
        loss = test_loss.item() / test_total.item()
        acc = test_correct.item() / test_total.item()
        print("Global Test Loss: {:.6f}; Test Accuracy: {:.2f}.\n".format(loss, acc))
        return loss, acc
    else:
        return None, None


def test(args, model_ps: DNNGoogleSpeechBatchNorm3LayerPS,
         model_prime: DNNGoogleSpeechBatchNorm3LayerModel,
         test_loader, epoch, test_loss_log, test_acc_log):
    # Do validation on every node in cluster.
    dispatch_model_for_validation(args, model_prime, model_ps)
    model_prime.update_model_from_buffer()
    model_prime.eval()
    test_loss = torch.zeros(1)
    test_correct = torch.zeros(1)
    test_total = torch.zeros(1)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % args.world_size == args.rank:
                data, target = batch['wav'].float(), batch['label']
                output = model_prime(data)
                test_loss += nn.functional.nll_loss(output, target, reduction='sum')  # sum up batch loss
                test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                test_correct += test_pred.eq(target.view_as(test_pred)).sum()
                test_total += target.shape[0]
    print("Epoch {} Local Test Loss: {:.6f}; Test Accuracy: {:.2f}.\n".format(epoch, test_loss.item()/test_total.item(),
                                                                              test_correct.item()/test_total.item()))
    if args.rank == 0:
        test_loss_log[epoch - 1], test_acc_log[epoch - 1] = sync_test_results(args, test_loss, test_correct, test_total)
    else:
        sync_test_results(args, test_loss, test_correct, test_total)


def worker_process(args):
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            rank=args.rank, world_size=args.world_size)
    device = torch.device('cpu')
    model_name = 'DNN_speech_3_layer_BN_' + str(args.epochs) + '_' + str(args.model_size) \
                 + '_subnet_' + str(args.world_size) + '_' + str(args.repartition_iter)
    train_set = speech_dataset.train_dataset()
    test_set = speech_dataset.test_dataset()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             drop_last=False)
    model_worker = DNNGoogleSpeechBatchNorm3LayerModel(args, model_size=args.model_size // args.world_size).to(device)
    model_prime = DNNGoogleSpeechBatchNorm3LayerModel(args, model_size=args.model_size).to(device)
    model_ps = DNNGoogleSpeechBatchNorm3LayerPS(rank=args.rank, partition_num=args.world_size,
                                                partitioned_model_size=args.model_size // args.world_size)
    optimizer = torch.optim.SGD(model_worker.parameters(), lr=args.lr)
    epochs = args.epochs * args.world_size
    train_time_log = np.zeros(epochs) if args.rank == 0 else None
    test_loss_log = np.zeros(epochs) if args.rank == 0 else None
    test_acc_log = np.zeros(epochs) if args.rank == 0 else None
    for epoch in range(1, epochs + 1):
        train(args, model_ps, model_worker, optimizer, train_loader, epoch, train_time_log)
        test(args, model_ps, model_prime, test_loader, epoch, test_loss_log, test_acc_log)
    if args.rank == 0:
        np.savetxt('./log/' + model_name + '_train_time.log', train_time_log, fmt='%1.4f', newline=' ')
        np.savetxt('./log/' + model_name + '_test_loss.log', test_loss_log, fmt='%1.4f', newline=' ')
        np.savetxt('./log/' + model_name + '_test_acc.log', test_acc_log, fmt='%1.4f', newline=' ')
        torch.save(model_prime.state_dict(), './trained_models/' + model_name + '.pth')


def main():
    parser = argparse.ArgumentParser(description='PyTorch 3-layer NN on google speech dataset (subnet distributed)')
    parser.add_argument('--dist-backend', type=str, default='gloo', metavar='S',
                        help='backend type for distributed PyTorch')
    parser.add_argument('--dist-url', type=str, default='tcp://127.0.0.1:9000', metavar='S',
                        help='master ip for distributed PyTorch')
    parser.add_argument('--rank', type=int, default=1, metavar='R',
                        help='rank for distributed PyTorch')
    parser.add_argument('--world-size', type=int, default=2, metavar='D',
                        help='partition group (default: 2)')
    parser.add_argument('--model-size', type=int, default=4096, metavar='N',
                        help='model size for intermediate layers (default: 4096)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--repartition-iter', type=int, default=30, metavar='N',
                        help='keep model in local update mode for how many iteration (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001 for BN)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    seed(0)  # This makes sure, node use the same random key so that they does not need to sync partition info.
    torch.manual_seed(args.seed)
    worker_process(args)


if __name__ == '__main__':
    main()
