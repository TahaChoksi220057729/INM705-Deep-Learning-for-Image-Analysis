import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def list_image_files(folder):
    folder = Path(folder)

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    return sorted(files)


def match_image_mask_pairs(images_dir, masks_dir):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    image_files = list_image_files(images_dir)

    if len(image_files) == 0:
        raise ValueError(f"No image files found in: {images_dir}")

    mask_by_stem = {}

    for mask_path in list_image_files(masks_dir):
        mask_by_stem[mask_path.stem] = mask_path

    pairs = []

    for image_path in image_files:
        mask_path = mask_by_stem.get(image_path.stem)

        if mask_path is None:
            raise FileNotFoundError(
                f"Mask file not found for image stem '{image_path.stem}' in {masks_dir}"
            )

        pairs.append((image_path, mask_path))

    return pairs


class SegmentationDataset(Dataset):
    def __init__(self, pairs, image_size, train=False, use_augmentation=True):
        self.pairs = list(pairs)
        self.image_size = image_size
        self.train = train
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.pairs)

    def augment(self, image, mask):
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < 0.2:
            angle = random.uniform(-20.0, 20.0)

            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR
            )

            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST
            )

        return image, mask

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR
        )

        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST
        )

        if self.train and self.use_augmentation:
            image, mask = self.augment(image, mask)

        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask


def resolve_dataset_root(data_dir):
    root = Path(data_dir)

    direct_train = root / "train" / "images"
    nested_train = root / "segmentation_task" / "train" / "images"

    if direct_train.is_dir():
        return root

    if nested_train.is_dir():
        return root / "segmentation_task"

    raise FileNotFoundError(
        "Could not find dataset structure. Expected either:\n"
        "data_dir/train/images and data_dir/train/masks\n"
        "or\n"
        "data_dir/segmentation_task/train/images and data_dir/segmentation_task/train/masks"
    )


def build_brisc_dataloaders(
    data_dir,
    image_size=256,
    batch_size=8,
    num_workers=2,
    use_augmentation=True
):
    root = resolve_dataset_root(data_dir)

    train_images_dir = root / "train" / "images"
    train_masks_dir = root / "train" / "masks"
    test_images_dir = root / "test" / "images"
    test_masks_dir = root / "test" / "masks"

    required_dirs = [
        train_images_dir,
        train_masks_dir,
        test_images_dir,
        test_masks_dir
    ]

    for directory in required_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing required directory: {directory}")

    train_pairs = match_image_mask_pairs(train_images_dir, train_masks_dir)
    val_pairs = match_image_mask_pairs(test_images_dir, test_masks_dir)

    train_dataset = SegmentationDataset(
        train_pairs,
        image_size=image_size,
        train=True,
        use_augmentation=use_augmentation
    )

    val_dataset = SegmentationDataset(
        val_pairs,
        image_size=image_size,
        train=False,
        use_augmentation=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset, val_dataset