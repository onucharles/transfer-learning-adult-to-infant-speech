import fire
from sc_01_save_embeddings import save_embeddings
from speech_commands_01_train_and_evaluate import train_and_evaluate as sc_train_eval
from voxceleb_01_train_and_evaluate import train_and_evaluate as vx_train_eval
from chillanto_01_train_and_evaluate import train_and_evaluate as chill_train_eval
from chillanto_speech_commands_02_transfer import model_transfer as chill_sc_transfer

if __name__ == '__main__':
    fire.Fire()
