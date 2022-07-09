from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import pandas as pd







Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255

        mask[mask > 0.5] = 2
        mask[mask <= 0.5] = 1
        mask[mask == 2] = 0

    return (img,mask)
def adjustData1(img,mask,mask1,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    else:
        img = img / 255
        mask = mask /255
        mask1 = mask1 / 255
        mask1[mask1 > 0.5] = 1
        mask1[mask1 <= 0.5] = 0


        mask[mask> 0.5]= 2
        mask[mask <= 0.5]= 1
        mask[mask == 2] = 0
    return (img,mask,mask1)


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    n=0
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
       # n=n+1
       # print('\n%d'%n)
        yield (img,mask)

def mutliGenerator(batch_size,train_path,image_folder,mask_folder,mask1_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    n=0
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    mask1_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    mask1_generator = mask1_datagen.flow_from_directory(
        train_path,
        classes=[mask1_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator, mask1_generator)
    for (img,mask,mask1) in train_generator:
        img,mask,mask1 = adjustData1(img,mask,mask1,flag_multi_class,num_class)
       # n=n+1
       # print('\n%d'%n)
        yield (img,[mask,mask1])

def mutliVGenerator(batch_size,train_path,image_folder,mask_folder,mask1_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    n=0
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    mask1_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    mask1_generator = mask1_datagen.flow_from_directory(
        train_path,
        classes=[mask1_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator, mask1_generator)
    for (img,mask,mask1) in train_generator:
        img,mask,mask1 = adjustData1(img,mask,mask1,flag_multi_class,num_class)
       # n=n+1
       # print('\n%d'%n)
        yield (img,[mask,mask1])
def testGenerator(test_path,num_image = 29,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        i1=i+1
        img = io.imread(os.path.join(test_path,"%03d.bmp"%i1),as_gray = as_gray)
        img = img / 255
       # img = trans.resize(img,target_size)

        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class and  as_gray) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img



def vaildGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:

        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        # n=n+1
        # print('\n%d'%n)

        yield (img, mask)

def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img

    img_out = np.zeros(img.shape + (3,))

    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        i1=i+1
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #img [img  > 0.5] = 1
        #img [img  <= 0.5] = 0

        img = img_as_ubyte(img)
        #img[img >=  1] = 255
        #img[img <1 ] = 0
        img=img*255
        io.imsave(os.path.join(save_path,"%02d_predict.png"%i1),img)
def saveResultb(save_path,npyfile,flag_multi_class = False,num_class = 2):
 for j in [0,1]:
    for i,item in enumerate(npyfile[j]):
        i1=i+1
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #img [img  > 0.5] = 1
        #img [img  <= 0.5] = 0

        img = img_as_ubyte(img)
        #img[img >=  1] = 255
        #img[img <1 ] = 0
        img=img*255
        io.imsave(os.path.join(save_path,"%02d_%d_predict.png"%(i1 ,j+1)),img)

def lossshow(loss):
    R = loss.history
    fig = plt.figure()

    x1 = loss.epoch
    x1 = [x11 + 1 for x11 in x1]
    x1 = list(map(str, x1))  # 使用list（map（str，x1））方法，将返回一个列表，列表中所有元素是str类型
    y = list(R.values())
    # y1=list(map(list,zip(*y1)) )
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    y4 = y[3]
    #y5 = y[4]
    #y6 = y[5]

        # 列表



    df = pd.DataFrame(y1, columns=['y1'])
    df.to_excel("y1.xlsx", index=False)

    df = pd.DataFrame(y2, columns=['y2'])
    df.to_excel("y2.xlsx", index=False)

    df = pd.DataFrame(y3, columns=['y3'])
    df.to_excel("y3.xlsx", index=False)

    df = pd.DataFrame(y4, columns=['y4'])
    df.to_excel("y4.xlsx", index=False)

    #df = pd.DataFrame(y5, columns=['y5'])
    #df.to_excel("y5.xlsx", index=False)

    #df = pd.DataFrame(y6, columns=['y6'])
    #df.to_excel("y6.xlsx", index=False)

   # print(y1)
    labe = list(R.keys())
    plt.plot(x1, y1, label=labe[0])  # label为设置图例标签，需要配合legend（）函数才能显示出
    plt.plot(x1, y2, label=labe[1])
    plt.plot(x1, y3, label=labe[2])
    plt.plot(x1, y4, label=labe[3])
    #plt.plot(x1, y5, label=labe[4])
    #plt.plot(x1, y6, label=labe[5])
    # 把x轴的刻度间隔设置为1，并存在变量里
    #plt.plot(x1, y5, label=labe[4])
    #plt.plot(x1, y6, label=labe[5])
    plt.xlabel('epochs')
    plt.ylabel('loss/acc')
    plt.title('train')
    plt.legend()  # 需要配合这个才能显示图例标签
    plt.show()

