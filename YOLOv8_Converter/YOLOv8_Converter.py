import pathlib
import shutil
from zipfile import ZipFile
import yaml


class YOLODataset:

    def __init__(self, dataset_path, overwrite=False):
        """
        Label Studio YOLO -> valid YOLOv8
        :param dataset_path: Path to dataset. Can be a folder (if you are preparing images) or a zip file (if you are converting a dataset)
        :param overwrite: Rewrite the dataset in the out folder
        """
        self._in_dataset = pathlib.Path(dataset_path)
        self._overwrite = overwrite
        self._init_folders()

    def _init_folders(self):
        self._datasets_path = pathlib.Path('datasets')
        self._cache_path = self._datasets_path / 'cache'
        self._temp_path = self._datasets_path / 'temp'

        self._cache_path.mkdir(parents=True, exist_ok=True)

        if self._temp_path.exists():
            shutil.rmtree(self._temp_path)

        self._temp_path.mkdir()

    def convert(self, val_perc=30, test_perc=0, absolute_path=False):
        """
        Convert Label Studio labeled dataset exported as YOLO to valid YOLOv8 format.
        :param val_perc: *Percentage of images used for validation*
        :param test_perc: *Percentage of images used for test*
        :param absolute_path: *Write full paths in data.yaml*
        :return: **PosixPath** *to data.yaml if success,* **Exception** *if err*
        """

        # CHECK SPLIT VALUES
        if val_perc + test_perc >= 100:
            return 'No images to train.'
        if val_perc + test_perc >= 50:
            print('WARNING: Train images perc <= 50%')

        # CHECK INPUT DATASET
        if not self._in_dataset.exists():
            return 'Dataset Path is not exist'
        if not self._in_dataset.is_file():
            return 'Dataset is not a file'
        if self._in_dataset.suffix != '.zip':
            return 'Dataset is not a zip file'

        # UNZIP DATASET
        try:
            with ZipFile(self._in_dataset, 'r') as f:
                f.extractall(self._temp_path)
        except Exception as e:
            return e

        # LOAD CLASSES
        try:
            with open(self._temp_path / 'classes.txt') as f:
                classes = {_id: _class for _id, _class in enumerate(f.read().split())}
        except Exception as e:
            return e

        # LOAD IMAGES
        in_images = [file for file in (self._temp_path / 'images').iterdir() if file.suffix in ['.png', '.jpg', '.jpeg']]
        in_labels = [file for file in (self._temp_path / 'labels').iterdir() if file.suffix in ['.txt']]

        # CHECK IF IMAGE OR LABEL UNPAIR
        unpaired_images = [file for file in in_images if file.stem not in [file.stem for file in in_labels]]
        unpaired_labels = [file for file in in_labels if file.stem not in [file.stem for file in in_images]]

        if unpaired_images or unpaired_labels:
            if len(unpaired_images) == len(in_images):
                print('CRITICAl:')
                return
            if len(unpaired_labels) == len(in_labels):
                print('CRITICAl:')
                return
            print(f'WARNING: Skipped {len(unpaired_images)} images and {len(unpaired_labels)} labels because them unpair.')

        # DELETE UNPAIR FILES FROM MAIN LIST
        in_images = [file for file in in_images if file not in unpaired_images]
        in_labels = [file for file in in_labels if file not in unpaired_labels]

        # SORT LABELS
        temp = []
        for image in in_images:
            for label in in_labels:
                if image.stem == label.stem:
                    temp.append(label)
        in_labels = temp; del temp

        # CALCULATE IMAGES
        val_c = int(len(in_images) * (val_perc / 100))
        test_c = int(len(in_images) * (test_perc / 100))
        train_c = len(in_images) - val_c - test_c

        # SPLIT AND JOIN LIST'S

        train_batch = in_images[:train_c]; train_batch.extend(in_labels[:train_c])
        val_batch = in_images[train_c:train_c + val_c]; val_batch.extend(in_labels[train_c:train_c + val_c])
        test_batch = in_images[train_c + val_c:]; test_batch.extend(in_labels[train_c + val_c:])

        files = (
            train_batch,
            val_batch,
            test_batch
        )

        # OUT PATH
        out_path = self._cache_path / self._in_dataset.stem
        if self._overwrite:
            if out_path.exists():
                shutil.rmtree(out_path)
        try:
            out_path.mkdir()
        except Exception as e:
            return e

        # SAVE FILES
        for _id, batch in enumerate(files):
            folder = out_path / ('train' if _id == 0 else 'val' if _id == 1 else 'test')
            if not batch: continue
            folder.mkdir()
            for file in batch:
                try:
                    shutil.copy(file, folder / file.name)
                except Exception as e:
                    return e

        # CONFIGURE DATA.YAML
        data = {
            'path': str(out_path) if not absolute_path else str(out_path.absolute()),
            'train': str(out_path / 'train') if not absolute_path else str((out_path / 'train').absolute()),
            'val': str(out_path / 'val') if not absolute_path else str((out_path / 'val').absolute()),
            'test': str(out_path / 'test') if not absolute_path else str((out_path / 'test').absolute()),
            'nc': len(classes),
            'names': classes
        }
        if not files[2]: del data['test']

        try:
            with open(out_path / 'data.yaml', 'w') as f:
                yaml.dump(data, f, sort_keys=False)
        except Exception as e:
            return e

        return (out_path / 'data.yaml').absolute() if absolute_path else out_path / 'data.yaml'

    def prepare(self, processor=None, split_100=False):
        """
        Prepare images before labeling in Label Studio
        :param processor: *Function to process images. Must accept* **PosixPath** *and return* **PIL.Image**
        :param split_100: *Write 100 images per folder*
        :return: **PosixPath** *to the location of the prepared images*
        """
        # CHECK FOLDER
        if not self._in_dataset.is_dir():
            return 'Dataset path is not a dir'
        if not list(self._in_dataset.iterdir()):
            return 'Dataset folder empty. Nothing to convert.'

        in_images = [file for file in self._in_dataset.iterdir() if file.suffix in ['.png', '.jpg', '.jpeg']]

        # COPY (or convert) IMAGES TO OUT FOLDER
        out_path = self._cache_path / ('prepare_' + self._in_dataset.stem)
        if self._overwrite:
            if out_path.exists():
                shutil.rmtree(out_path)
        try:
            out_path.mkdir()
        except Exception as e:
            return e

        split_folder = None

        for _id, image in enumerate(in_images):
            if split_100:
                split_folder = str(_id // 100)
                (out_path / split_folder).mkdir(exist_ok=True)
            try:
                if processor:
                    file = processor(image)
                    file.save(((out_path / split_folder) if split_100 else out_path) / f'{_id}{image.suffix}')
                else:
                    shutil.copy(image, ((out_path / split_folder) if split_100 else out_path) / f'{_id}{image.suffix}')
            except Exception as e:
                return e

        return out_path
