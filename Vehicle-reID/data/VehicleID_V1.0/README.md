# VehicleID
-------------------------------------------------------------------------------------------
## Copyright and Citation
Copyright (c) 2016 National Engineering Laboratory for Video Technology  (NELVT)
at Peking University in Beijing, P.R.China.

This dataset is intended for research purposes only and as such cannot be used commercially.
If you find this dataset useful in your research works, please consider citing:

    @inproceedings{liu2016deep,
        title={Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles},
        author={Liu, Hongye and Tian, Yonghong and Wang, Yaowei and Pang, Lu and Huang, Tiejun},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        pages={2167--2175},
        year={2016}
    }

-------------------------------------------------------------------------------------------
## Description
This dataset is organized as follows:
```


0 directories, 7 files
.
├── attribute
│   ├── model_names.txt                 # model ID: 250 vehicle models encoded from 0 to 249
│   ├── color_names.txt                 # color ID: 7 colors encoded from 0 to 6
│   ├── img2vid.txt                     # image name(without .jpg file extension) -> vehicle ID
│   ├── model_attr.txt                  # vehicle ID -> model ID
│   └── color_attr.txt                  # vehicle ID -> color ID
├── train_test_split
│   └── train_list.txt                  # image list (13164 vehicles) for model training
│   ├── test_list_800.txt               # image list (800 vehicles) for model testing(small test set in paper)
│   ├── test_list_1600.txt              # image list (1600 vehicles) for model testing(medium test set in paper)
│   ├── test_list_2400.txt              # image list (2400 vehicles) for model testing(large test set in paper)
│   ├── test_list_3200.txt              # image list (3200 vehicles) for model testing
│   ├── test_list_6000.txt              # image list (6000 vehicles) for model testing
│   ├── test_list_13164.txt             # image list (13164 vehicles) for model testing
├── image
│   ├── *.jpg                           # vehicle images captured from real-world surveillance cameras
├── README.md                           # this readme file containing copyright info and dataset usage
```
1. "img2vid.txt" includes all vehicle images in this dataset and each image is attached with an integer as their identity info(vid is short for vehicle ID). All together, there are 221567 images of 26328 vehicles(8.42 images/vehicle in average).
2. "model_attr.txt" includes all vehicles which have been manually labeled with their vehicle model info(e.g. Audi A6L 2012).
3. "color_attr.txt" includes all vehicles which have been manually labeled with their color info(e.g. black). Notice that vehicles with color info also have their model info("color_attr.txt" is a subset of "model_attr.txt").
4. "color_names.txt" and "model_names.txt" contain the original names of the color and vehicle model labels. For instance, red color is encoded as 3 and Audi A6L is encoded as 2. Since all images are collected in a city in China and the vehicle model info is also labeled by people from China, model names in "model_names.txt" are in Chinese.

## News
There are a few differences between the info of this readme file and the info in our paper("Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles"):
 1. We only have vehicle ID and vehicle model when writing the paper. The color info was added after the paper submission(thus, the color info has not been used in our experiments) and we believe it can benefit the future research works on vehicle search related problems.
 2. The image number and vehicle number are slightly different than the numbers in our paper. The reason is the vehicle IDs are originally encoded by the license numbers and we found some fake plate vehicles recently. That means a "Volkswagen Lavida" and a "Chevrolet Cruze" may be labeled with the same vehicle ID if one faked the other's license plate. We then checked our dataset again, updated the "img2vid.txt" file and the corresponding train-test split files("train\_list.txt", "test\_list\_\*.txt").

-------------------------------------------------------------------------------------------




