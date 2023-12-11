# YOLOv8_Converter
#### Small script for converting/training a YOLOv8 model on a dataset labeled in Label Studio

## ⏬ How to install?
>  Just type to terminal `pip install git+https://github.com/wh0us/YOLOv8_Converter/`

## ⚙️ How to use?
> ## First you need to get dataset!
> #### Go to Label Studio and export your dataset as YOLO dataset
>
> ![Export](https://github.com/wh0us/YOLOv8_Converter/blob/main/export.png)

> ### Use from terminal
> #### ❌ Currently not working.
> #### Convert dataset
> `ydataset --path PATH-TO-DATASET --val VAL-BATCH-PERC --test TEST-BATCH-PERC`
> #### Convert dataset with custom path
> `ydataset --path PATH-TO-DATASET --out-path OUT-PATH --val VAL-BATCH-PERC --test TEST-BATCH-PERC`

> ### Use in code
> ```python
>from YOLOv8_Converter import YOLODataset
>
># convert exported dataset
>converter = YOLODataset('data.zip')
>converted = converter.convert(val_perc=30, test_perc=0, absolute_path=False)
>
># prepare images to labeling
>converter = YOLODataset('image folder')
>converted = converter.prepare(processor=None, split_100=False)


# 🎉🎉 ENJOY!
### With ❤️‍🔥 from wh0us
