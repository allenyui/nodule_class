from ImageAugmentation import DataAug3D

aug = DataAug3D(rotation=45, width_shift=0.05, height_shift=0.05, depth_shift=0, zoom_range=0)
aug.DataAugmentation('Nodule.csv', 40, aug_path='E:\Workplace\dataset\\classification_aug1\\')
