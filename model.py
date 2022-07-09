import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(outputs=[conv10], inputs=[inputs])

    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def nestunet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    con2up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv2))
    cot11 = concatenate([conv1, con2up], axis=3)
    # 卷积
    cot11 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot11)
    cot11 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot11)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    con3up = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv3))
    cot21 = concatenate([conv2, con3up], axis=3)
    # 卷积
    cot21 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot21)
    cot21 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot21)
    cot21up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(cot21))
    cot12 = concatenate([conv1,cot11, cot21up], axis=3)
    # 卷积
    cot12 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot12)
    cot12 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot12)
    out1 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot12)
    out1 = Conv2D(1, 1, activation='sigmoid', name="out1")(out1)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    con4up = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop4))
    cot31 = concatenate([conv3, con4up], axis=3)
    # 卷积
    cot31 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot31)
    cot31 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot31)
    cot31up = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(cot31))
    cot22 = concatenate([conv2,cot21, cot31up], axis=3)
    # 卷积

    cot22 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot22)
    cot22 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot22)
    cot22up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(cot22))
    cot13 = concatenate([conv1,cot11,cot12, cot22up], axis=3)
    # 卷积
    cot13 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot13)
    cot13 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot13)


    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    con5up = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    cot41 = concatenate([drop4, con5up], axis=3)
    # 卷积
    cot41 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot41)
    cot41 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot41)
    cot41up = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(cot41))
    cot32 = concatenate([conv3,cot31, cot41up], axis=3)
    # 卷积
    cot32 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot32)
    cot32 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot32)
    cot32up = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(cot32))
    cot23 = concatenate([conv2,cot21,cot22, cot32up], axis=3)
    # 卷积
    cot23 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot23)
    cot23 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot23)
    cot23up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(cot23))
    cot14 = concatenate([conv1,out1,cot12,cot13, cot23up], axis=3)#conv1,cot11,
    # 卷积
    cot14 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot14)
    cot14 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot14)
    cot14 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot14)
    conv10 = Conv2D(1, 1, activation='sigmoid',name = "out2")(cot14)

    model = Model(outputs=[out1,conv10], inputs=[inputs])

    model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy','binary_crossentropy'], metrics=['accuracy','accuracy'],loss_weights={"out2":0.5,"out1":0.5})





    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)



    return model

def unetplus(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    con2up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv2))
    cot11 = concatenate([conv1, con2up], axis=3)
    # 卷积
    cot11 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot11)
    cot11 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot11)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    con3up = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv3))
    cot21 = concatenate([conv2, con3up], axis=3)
    # 卷积
    cot21 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot21)
    cot21 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot21)
    cot21up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(cot21))
    cot12 = concatenate([conv1,cot11, cot21up], axis=3)
    # 卷积
    cot12 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot12)
    cot12 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot12)


    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    con4up = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop4))
    cot31 = concatenate([conv3, con4up], axis=3)
    # 卷积
    cot31 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot31)
    cot31 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot31)
    cot31up = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(cot31))
    cot22 = concatenate([conv2,cot21, cot31up], axis=3)
    # 卷积

    cot22 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot22)
    cot22 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot22)
    cot22up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(cot22))
    cot13 = concatenate([conv1,cot11,cot12, cot22up], axis=3)
    # 卷积
    cot13 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot13)
    cot13 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot13)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    con5up = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    cot41 = concatenate([drop4, con5up], axis=3)
    # 卷积
    cot41 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot41)
    cot41 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot41)
    cot41up = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(cot41))
    cot32 = concatenate([conv3,cot31, cot41up], axis=3)
    # 卷积
    cot32 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot32)
    cot32 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot32)
    cot32up = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(cot32))
    cot23 = concatenate([conv2,cot21,cot22, cot32up], axis=3)
    # 卷积
    cot23 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot23)
    cot23 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot23)
    cot23up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(cot23))
    cot14 = concatenate([conv1,cot11,cot12,cot13, cot23up], axis=3)#conv1,cot11,
    # 卷积
    cot14 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot14)
    cot14 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot14)
    cot14 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(cot14)
    conv10 = Conv2D(1, 1, activation='sigmoid',name = "out2")(cot14)

    model = Model(outputs=[conv10], inputs=[inputs])

    model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=['accuracy'])





    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)



    return model