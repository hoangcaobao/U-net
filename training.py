from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from preparedata import *
from model_structure import *

seed=2021
data_gen_args = dict(rotation_range=0.3,
                    width_shift_range=0.08,
                    height_shift_range=0.08,
                    shear_range=0.08,
                    zoom_range=0.08,
                    horizontal_flip=True,
                    fill_mode='nearest')

image_generator = generate_data(train_path="data",data_folder="image",data_type="image",aug_dict=data_gen_args, save_to_dir="data/aug", seed=seed)
mask_generator=generate_data(train_path="data",data_folder="label", data_type="mask",aug_dict=data_gen_args, save_to_dir="data/aug", seed=seed)
train=zip_data(image_generator,mask_generator)

model=unet('unet_weights.hdf5')
model_checkpoint=ModelCheckpoint('unet_weights.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(train, steps_per_epoch=300, epochs=5, callbacks=[model_checkpoint])

