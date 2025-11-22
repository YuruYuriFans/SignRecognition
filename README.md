# SignRecognition

## File structure
```bash
.
├── ablated_models
├── ablation.py
├── augmentation.py
├── clean.py
├── dataset.py
├── demo
│   └── tuning_demo.txt
├── evaluation
│   ├── factory.txt
│   ├── GTSRB_Test_GT.csv
├── evaluation.py
├── Final_Test
│   └── Images
├── Final_Training
│   └── Images
├── GTSRB_Test_GT.csv
├── model.py
├── predictions
├── predict.py
├── README.md
├── records
│   ├── ablation_results.csv
│   ├── data_augmentation.txt
│   ├── factory.txt
│   ├── log_predict.txt
│   ├── predicting_results.txt
│   ├── scaffold_test.txt
│   ├── statistics_explanation.txt
│   ├── train_results.txt
│   ├── train.txt
│   └── tune_results.txt
├── requirements.txt
├── results
│   ├── comparison_results_20251112_124045.json
├── testloader.py
├── test.py
├── trained_models
│   ├── best_lenet_basic.pth
│   ├── best_minivgg_basic.pth
│   ├── best_mobilenetv2_025_basic.pth
│   └── best_mobilenetv4_small_basic.pth
├── train.py
├── tree.txt
├── tuned_models
└── tune.py

```

There is no automatic log/model cleaning yet so the working directory may become increasing large and messy after several runs. `clean.py` can be manually used to  remove `.pth` files other than the 4 basic models.

## requirements.txt

```bash
pip install -r requirements.txt
```

or

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 numpy>=1.24.0 pandas>=1.5.0 Pillow>=9.0.0 matplotlib>=3.5.0 seaborn>=0.12.0 graphviz>=0.20.0 thop>=0.1.1.post2209072238 psutil>=5.9.0 torchinfo>=1.7.0 scikit-learn>=1.2.0 tqdm>=4.64.0
```

## Instructions to run the code
The report is based on the following order of commands.

To train the basic models and predict:

```bash
mkdir -p demo
python3 train.py
python3 predict.py > demo/basic_demo.txt
python3 clean.py
```

To demonstrate augmentation effect:

```bash
python train.py --aug none basic advanced --compare
python3 predict.py > demo/augmentation_demo.txt
python3 clean.py
```

To demonstrate ablation comparison:

```bash
python3 ablation.py
python3 predict.py > demo/ablation_demo.txt
python3 clean.py
```

To demonstrate tuning comparison

```bash
python3 tune.py
python3 predict.py > demo/tuning_demo.txt
python3 clean.py
```