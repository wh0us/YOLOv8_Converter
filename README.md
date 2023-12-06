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
> #### Convert dataset
> `ydataset --path PATH-TO-DATASET --val VAL-BATCH-PERC --test TEST-BATCH-PERC`
> #### Convert dataset with custom path
> `ydataset --path PATH-TO-DATASET --out-path OUT-PATH --val VAL-BATCH-PERC --test TEST-BATCH-PERC`

> ### Use in code
> ```python
> from YOLOv8_Converter import conf_logger, convert_dataset, update_dataset_pathes 
> conf_logger('TRACE')
> 
> path = 'path to your dataset'
> 
> convert_dataset(path, out_path=None, val_perc=30, test_perc=0, rename=None)
>
> update_dataset_pathes(path) # if you moved converted dataset in another dir

## ⏳ Future updates
> - add train interface
> - cleaning up
> - bug fixes


# 🎉🎉 ENJOY!
### With ❤️‍🔥 from wh0us
