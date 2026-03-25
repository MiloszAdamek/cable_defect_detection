from torch.utils.data import DataLoader
from src.utils import load_train_paths, CableDataset
from src.utils import DATA_PATH

train_paths = load_train_paths(DATA_PATH)

train_dataset = CableDataset(train_paths)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)