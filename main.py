from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import *
from data import *
from keras.callbacks import TensorBoard





#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')


myGene1 = mutliGenerator(1,'data/membrane/train','image','labelb','label',data_gen_args,save_to_dir = None,image_color_mode = "rgb")
myvaild1 =  mutliVGenerator(1,"data/membrane/vaild",'image',"labelb",'label',data_gen_args,save_to_dir = None,image_color_mode = "rgb")
lr=[1e-4,1e-5,1e-6]
'''for (img, mask) in myGene1:
    mask=mask[1].reshape(256,256)
    plt.imshow(mask)
    plt.show()
    i=1
'''

model = nestunet(input_size=(256, 256, 3))  # ,pretrained_weights='nestunet_membrane.h5', validation_data=myvaild,validation_steps=10
model_checkpoint = ModelCheckpoint('unetplus_back_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)



M = model.fit_generator(myGene1, steps_per_epoch=500, epochs=15, callbacks=[model_checkpoint],validation_data=myvaild1,validation_steps=27)


lossshow(M)


testGene = testGenerator("data/membrane/test/image",target_size = (256,256,3),as_gray = False)
model.save('unetplus_back_membrane.h5')
results = model.predict_generator(testGene,28,verbose=1)
saveResultb("data/membrane/test/unetplustest2",results)
