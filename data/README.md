## Dataset

We use the CIFAR-10 dataset, which consists of 60,000 32×32 color images across 10 classes. The dataset is downloaded via `torchvision.datasets.CIFAR10`.

## Preprocessing

Training images are augmented with random horizontal flips and random crops (32×32 with 4-pixel padding), then normalized using the CIFAR-10 channel means (0.4914, 0.4822, 0.4465) and standard deviations (0.2023, 0.1994, 0.2010). Test images are normalized without augmentation.

## Data splits

The original 50,000 training images are split into a training set and a validation set using `torch.utils.data.random_split` with a fixed random seed for reproducibility. The held-out 10,000 images serve as the test set. All splits are loaded via `DataLoader` with batched loading and pinned memory for GPU efficiency.

---

Our actual code to obtain the data:

```python
SEED = 42
BATCH_SIZE = 128

N_TRAIN = 45_000
N_VAL =  5_000

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),   # translation by up to 4px
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

full_train = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=train_tf)
test_set   = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)

train_set, val_set = random_split(
    full_train, [N_TRAIN, N_VAL],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_set,   batch_size=512, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
```
