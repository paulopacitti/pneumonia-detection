# pneumonia-detection
🫁 Deep learning model for automatic pneumonia detection in x-rays. Built with pytorch. Inspired by the article _[Identifying Medical Diagnoses and Treatable
Diseases by Image-Based Deep Learning](https://doi.org/10.1016/j.cell.2018.02.010)_.

![AI generated with DALL-E 3](https://raw.githubusercontent.com/paulopacitti/pneumonia-detection/main/docs/repo_cover.jpg)

## Dataset
Two datasets are used, one of exams of children from 0 to 5 years old of the _[Guangzhou Women and Children’s Medical Center](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)_, and other from the _[National Institutes of Health (NIH)](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset)_, consisting of chest x-rays of different diseases. It can be downloaded by running the `setup.py` script.
```bash
python setup.py
```

## Models
Models are stored in the `checkpoints/` directory. Usually, the most recent model is the one which is better trained and have better results.

## Contributing
All development should be done in the Python modules. The `main.ipynb` is the main experimentation notebook and it should only be used when the experimentations are decided to be part of the research, otherwise, use other notebooks for this. Notebooks are not commited, as defined in `.gitgnore`.