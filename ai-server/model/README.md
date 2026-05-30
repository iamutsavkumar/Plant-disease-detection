# model/

Place your trained model files here.

## Required files

| File                   | Description                                      |
|------------------------|--------------------------------------------------|
| `plantmd_model.keras`  | Trained Keras model (output of `train_model.py`) |
| `class_labels.json`    | Class index → label mapping (auto-generated)     |

## Training

```bash
cd ai-server
python train_model.py --data ./PlantVillage --epochs 20 --batch 32
```

The training script automatically saves both files to this directory.

## Mock mode

If neither file is present, the server starts in **mock mode** and returns
random (but realistic) predictions. This lets you develop and test the full
stack without a trained model.
