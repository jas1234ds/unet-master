from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(1,'data/membrane/train','image','labelR',data_gen_args,save_to_dir = None,image_color_mode = "rgb")
myvaild = vaildGenerator(1,"data/membrane/vaild",'image',"labelR",data_gen_args,save_to_dir = None,image_color_mode = "rgb")
lr=[1e-4,1e-5,1e-6]
'''for (img, mask) in myGene1:
    mask=mask[1].reshape(256,256)
    plt.imshow(mask)
    plt.show()
    i=1
'''

model = unetplus(input_size=(256, 256, 3))  # ,pretrained_weights='unet_back_membrane.h5', validation_data=myvaild,validation_steps=10
model_checkpoint = ModelCheckpoint('membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
M = model.fit_generator(myGene, steps_per_epoch=300, epochs=20, callbacks=[model_checkpoint], validation_data=myvaild,validation_steps=27)

lossshow(M)


testGene = testGenerator("data/membrane/test/image",target_size = (256,256,3),as_gray = False)
model.save('unet_back_membrane.h5')
results = model.predict_generator(testGene,28,verbose=1)
saveResult("data/membrane/test/unetplus",results)
