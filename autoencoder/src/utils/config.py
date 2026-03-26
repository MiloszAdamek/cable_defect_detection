from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_PATH = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models"

IMG_SIZE = 256
BATCH_SIZE = 32