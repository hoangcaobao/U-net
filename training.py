from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from preparedata import *
from model_structure import *
import os
seed=2021
data_gen_args = dict(rotation_range=0.3,
                    width_shift_range=0.08,
                    height_shift_range=0.08,
                    shear_range=0.08,
                    zoom_range=0.08,
                    horizontal_flip=True,
                    validation_split=0.2,
                    fill_mode='nearest')

train_image_generator = generate_data_train(train_path="data",data_folder="image",data_type="image",aug_dict=data_gen_args, save_to_dir="data/aug", seed=seed)
train_mask_generator=generate_data_train(train_path="data",data_folder="label", data_type="mask",aug_dict=data_gen_args, save_to_dir="data/aug", seed=seed)
train=zip_data(train_image_generator,train_mask_generator)
valid_image_generator = generate_data_valid(train_path="data",data_folder="image",data_type="image",aug_dict=data_gen_args,  save_to_dir="data/aug",seed=seed)
valid_mask_generator = generate_data_valid(train_path="data",data_folder="label", data_type="mask",aug_dict=data_gen_args,  save_to_dir="data/aug",seed=seed)
valid=zip_data(valid_image_generator,valid_mask_generator)    

model=unet('unet_weights.hdf5')
model_checkpoint=ModelCheckpoint('unet_weights.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(train, validation_data=valid, validation_steps=valid_image_generator.samples//32, steps_per_epoch=train_image_generator.samples//32, epochs=50, callbacks=[model_checkpoint])

