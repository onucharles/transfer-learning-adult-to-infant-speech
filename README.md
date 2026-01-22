# ubenwa-transfer-learning

Transfer learning of models trained on a variety of adult speech tasks to infant cry tasks

**Paper:** [Neural Transfer Learning for Cry-based Diagnosis of Perinatal Asphyxia](https://arxiv.org/pdf/1906.10199)

## Setup

### Environment Variables

This project uses Comet ML for experiment tracking. You must set the following environment variables before running any training or evaluation scripts:

```bash
export COMET_API_KEY="your-comet-api-key"
export COMET_REST_API_KEY="your-comet-rest-api-key"  # Required for notebooks
export COMET_WORKSPACE="your-workspace-name"
```

**Important Security Note:**
- Never commit API keys to version control
- Keep your API keys private and secure
- If you suspect your keys have been exposed, regenerate them immediately from the Comet ML dashboard

You can also set these in your IDE's environment settings or in a shell configuration file (e.g., `.bashrc`, `.zshrc`).

## Files
* `train.py` and `visualise.py`: Train and visualise neural models
* `train_classical.py` and `visualise_classical.py`: Train and visualise classical models (only SVM currently supported)
* `model.py`: Neural models and architectures
* `manage_audio.py`: Audio processing
* `ConfigBuilder.py`: Class for chaining and parsing config parameters.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{onu2019neural,
  title={Neural Transfer Learning for Cry-based Diagnosis of Perinatal Asphyxia},
  author={Onu, Charles C. and Lebensold, Jonathan and Hamilton, William L. and Precup, Doina},
  journal={arXiv preprint arXiv:1906.10199},
  year={2019}
}
```

**Paper:** Onu, C. C., Lebensold, J., Hamilton, W. L., & Precup, D. (2019). Neural Transfer Learning for Cry-based Diagnosis of Perinatal Asphyxia. arXiv preprint arXiv:1906.10199. Available at: https://arxiv.org/pdf/1906.10199