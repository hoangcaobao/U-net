import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def preprocessing_data(image,mask):
    image=np.array(image)
    mask=np.array(mask)
    image=image/255
    mask=mask/255
    mask[mask<=0.5]=0
    mask[mask>0.5]=0
    return image,mask

def generate_data_valid(train_path, data_folder, aug_dict, data_type=None,save_to_dir=None, seed=1):
    datagen=ImageDataGenerator(**aug_dict)
    data_generator=datagen.flow_from_directory(
        train_path,
        classes=[data_folder],
        class_mode=None,
        batch_size=2,
        color_mode="grayscale",
        target_size=[256, 256],
        save_to_dir=save_to_dir,
        save_prefix=data_type,
        subset="validation",
        seed=seed
    )
    return data_generator

def generate_data_train(train_path, data_folder, aug_dict, data_type=None,save_to_dir=None, seed=1):
    datagen=ImageDataGenerator(**aug_dict)
    data_generator=datagen.flow_from_directory(
        train_path,
        classes=[data_folder],
        class_mode=None,
        color_mode="grayscale",
        target_size=[256, 256],
        save_to_dir=save_to_dir,
        save_prefix=data_type,
        subset="training",
        seed=seed
    )
    return data_generator

def zip_data(image_generator, mask_generator):
    train_generator=zip(image_generator, mask_generator)
    for (image, mask) in train_generator:
        image,mask=preprocessing_data(image, mask)
        yield(image, mask)
