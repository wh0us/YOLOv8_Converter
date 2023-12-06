import argparse
from pathlib import Path
from loguru import logger
from sys import stdout
from shutil import rmtree, copy
from os import rename as rn
from zipfile import ZipFile
import yaml
logger.remove()

def convert_dataset(path, out_path=None, val_perc=30, test_perc=0, rename=None):
    if val_perc + test_perc >= 100:
        logger.critical('Sum of val and test cannot be more than 100%'); exit()
    # ---------------------- FOLDERS CHECK ----------------------
    logger.trace('Checking folders')
    # -------  TEMP PATH -------
    temp_path = Path('datasets/temp')
    if temp_path.exists() and list(temp_path.iterdir()): rmtree(temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)
    # ------ DATASET PATH ------
    if not Path(path).exists():
        logger.critical('The path doesn\'t exist.')
        exit()
    if not Path(path).is_file():
        logger.critical('The path must be to the .zip file.')
        exit()
    else:
        path = Path(path).absolute()
    # ---- DATASET OUT PATH ----
    if out_path:
        (Path(out_path) / path.stem).mkdir(parents=True, exist_ok=True)
        out_path = (Path(out_path) / path.stem).absolute()
    else:
        out_path = Path(f'datasets/cache/{path.stem}')
        out_path.mkdir(parents=True, exist_ok=True)
    if list(out_path.iterdir()):
        logger.critical('Out path folder is not empty!'); exit()
    logger.trace('All necessary folders exist.')
    # ---------------------- UNZIP DATASET ----------------------
    logger.trace('Unzipping dataset.')

    with ZipFile(path, 'r') as zip:
        zip.extractall(path=temp_path)

    logger.trace('Dataset unzipped.')
    # ------------------ LOAD DATA FROM DATASET -----------------
    logger.trace('Loading data from dataset.')
    with open(temp_path / 'classes.txt', 'r') as f:
        dataset_classes = {_id: _class for _id, _class in enumerate(f.read().split())}
        dataset_classes_count = len(dataset_classes)
    dataset_images = sorted((temp_path / 'images').iterdir(), key=lambda file: int(file.stem.split('-')[1]))
    dataset_labels = sorted((temp_path / 'labels').iterdir(), key=lambda file: int(file.stem.split('-')[1]))
    logger.trace('Data loaded.')
    logger.trace('Check files count.')
    if len(dataset_images) != len(dataset_labels):
        logger.critical('The number of images and labels do not match!'); exit()
    # ------------ CALCULATE TRAIN, VAL, TEST IMAGES ------------
    logger.trace('Calculating split sizes.')
    val_count = int(len(dataset_images) * (val_perc / 100))
    test_count = int(len(dataset_images) * (test_perc / 100))
    train_count = len(dataset_images) - val_count - test_count
    # ----------------------- SPLIT IMAGES ----------------------
    # --------- IMAGES ---------
    train_images = dataset_images[:train_count]
    val_images = dataset_images[train_count:train_count+val_count]
    test_images = dataset_images[train_count+val_count:]
    # --------- LABELS ---------
    train_labels = dataset_labels[:train_count]
    val_labels = dataset_labels[train_count:train_count + val_count]
    test_labels = dataset_labels[train_count + val_count:]
    # --------- BATHES ---------
    image_batch = [train_images, val_images, test_images]
    label_batch = [train_labels, val_labels, test_labels]
    # ----------------------- RENAME IMAGES ---------------------
    if rename:
        logger.trace('Renaming images.')
        for batch_id, batch in enumerate(image_batch):
            if not len(batch): continue
            for file_id, file in enumerate(batch):
                rn(src=file, dst=file.parent / (f'image_{file_id}' + file.suffix))
                image_batch[batch_id][file_id] = file.parent / (f'image_{file_id}' + file.suffix)
        for batch_id, batch in enumerate(label_batch):
            if not len(batch): continue
            for file_id, file in enumerate(batch):
                rn(src=file, dst=file.parent / (f'label_{file_id}' + file.suffix))
                label_batch[batch_id][file_id] = file.parent / (f'label_{file_id}' + file.suffix)
        logger.trace('Renamed.')
    # ------------------------ WRITE DATA -----------------------
    # ----- PREPARE  DATA ------
    images = {'train': image_batch[0], 'val': image_batch[1], 'test': image_batch[2]}
    labels = {'train': label_batch[0], 'val': label_batch[1], 'test': label_batch[2]}
    # --------- WRITE ----------
    logger.trace('Write data.')
    for batch in [images, labels]:
        for path, files in batch.items():
            if not len(files): continue
            path = out_path / path
            path.mkdir(exist_ok=True)
            for file in files:
                copy(file, path)
    dataframe = {
        'path': str(out_path.absolute()),
        'train': str((out_path / 'train').absolute()),
        'val': str((out_path / 'val').absolute()),
        'test': str((out_path / 'test').absolute()),
        'nc': dataset_classes_count,
        'names': dataset_classes
    }
    if not test_count: del dataframe['test']
    with open(out_path / 'data.yaml', 'w') as f:
        yaml.dump(dataframe, f, sort_keys=False)
    logger.info(f'Images used for training: {len(train_images)} - {100 - val_perc - test_perc}%')
    logger.info(f'Images used for validating: {len(val_images)} - {val_perc}%')
    logger.info(f'Images used for test: {len(test_images)} - {test_perc}%')
    logger.success(f'Dataset converted! Path to data.yaml: {out_path / "data.yaml"}')
    logger.success(f'Full path to data.yaml: {(out_path / "data.yaml").absolute()}')
    # ----- DEL TEMP PATH ------
    rmtree(temp_path)

def update_dataset_pathes(path):
    logger.trace('Checking folder.')
    if not Path(path).exists():
        logger.critical('The path doesn\'t exist.'); exit()
    if not Path(path).is_dir():
        logger.critical('The path must be already converted dataset folder.'); exit()
    logger.trace('Checked.')
    logger.trace('Loading data.yaml.')
    dataset_new_path = Path(path).absolute()
    with open(dataset_new_path / 'data.yaml', 'r') as f:
        dataframe = yaml.safe_load(f)
    logger.trace('Loaded.')
    logger.trace('Editing.')
    dataframe['path'] = str(dataset_new_path)
    dataframe['train'] = str(dataset_new_path / 'train')
    dataframe['val'] = str(dataset_new_path / 'val')
    try:
        dataframe['test']
        dataframe['test'] = str(dataset_new_path / 'test')
    except: pass
    logger.trace('Edited.')
    logger.trace('Writing to file.')
    with open(dataset_new_path / 'data.yaml', 'w') as f:
        yaml.dump(dataframe, f, sort_keys=False)
    logger.trace('Writed.')
    logger.success('Pathes updated!')
def main():
    parser = argparse.ArgumentParser(
        prog='ydataset',
        description='Small script for converting/training a YOLOv8 model on a dataset labeled in Label Studio',
        epilog='More information can be found in the git repository: {git}')

    parser.add_argument('--path', type=str, help='Path to your dataset.zip', required=True)
    parser.add_argument('--out-path', type=str, help='Path to out converted dataset')
    parser.add_argument('--update-pathes', help='Update pathes in dataset (if moved)', action='store_true')
    parser.add_argument('--val', type=int, help='Percentage of images used for validation', default=30)
    parser.add_argument('--test', type=int, help='Percentage of images used for test', default=0)
    parser.add_argument('--rename', help='Rename images & txt to format image_001.png/label_001.txt', action='store_true')
    parser.add_argument('--verbose', help='On/Off logging', action='store_true')
    args = parser.parse_args()
    logging = args.verbose
    logger.add(stdout, format='[<fg #7f25f5>Dataset-Configurator</fg #7f25f5>] [<fg #ac25f5>{time:HH.mm.ss}</fg #ac25f5>] <lvl>{message}</lvl>', level='TRACE' if logging else 'INFO')

    if not args.update_pathes:
        convert_dataset(args.path, args.out_path, args.val, args.test, args.rename)
    else:
        update_dataset_pathes(args.path)


if __name__ == '__main__':
    main()

def conf_logger(level):
    logger.add(stdout, format='[<fg #7f25f5>Dataset-Configurator</fg #7f25f5>] [<fg #ac25f5>{time:HH.mm.ss}</fg #ac25f5>] <lvl>{message}</lvl>', level=level)
