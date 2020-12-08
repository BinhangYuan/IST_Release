wget https://s3.amazonaws.com/binhang/data_speech_commands_v0.02.tar.gz

mkdir data_speech_commands_v0.02

tar -xvf data_speech_commands_v0.02.tar.gz -C data_speech_commands_v0.02/

pip install librosa

python google_speech_data_loader.py

mkdir log

mkdir trained_models

mkdir figures

rm data_speech_commands_v0.02.tar.gz
