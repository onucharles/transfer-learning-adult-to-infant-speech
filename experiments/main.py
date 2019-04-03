import fire
from sc_01_save_embeddings import save_embeddings
from speech_commands_01_train_and_evaluate import train_and_evaluate as sc_train_eval
from voxceleb_01_train_and_evaluate import train_and_evaluate as vx_train_eval
from chillanto_01_train_and_evaluate import train_and_evaluate as chill_train_eval
from audioset_01_train_and_evaluate import train_and_evaluate as audioset_train_eval
from vctk_train_and_evaluate import train_and_evaluate as vctk_train_eval
from esc50_train_and_evaluate import train_and_evaluate as esc50_train_eval
from sitw_train_and_evaluate import train_and_evaluate as sitw_train_eval

# Transfer Learning
from chillanto_02_transfer import model_transfer as chill_transfer

# Noise Experiments
# from noise_evaluate import noise_evaluate as noise_eval

# Ablation Experiments
# from freq_ablation_evaluate import freq_ablation_evaluate as freqab_eval

# Timeshift Experiments
# from timeshift_evaluate import timeshift_evaluate as timeshift_eval

# Save embeddings experiment
from chillanto_save_embeddings import save_embeddings as save_emb

# Debug Experiments
from chillanto_sc_debug import sc_debug as sc_debug
if __name__ == '__main__':
    fire.Fire()
