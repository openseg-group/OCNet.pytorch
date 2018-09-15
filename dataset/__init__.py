from .cityscapes import CitySegmentationTrain, CitySegmentationTest, CitySegmentationTrainWpath

datasets = {
	'cityscapes_train': CitySegmentationTrain,
	'cityscapes_test': CitySegmentationTest,
	'cityscapes_train_w_path': CitySegmentationTrainWpath,
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)