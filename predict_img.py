from model import *
from data import *
model = tf.keras.models.load_model('unetplus_back_membrane.h5', compile=True) #  注意这儿得compile需要设置为true，如果你不设置你需要多一步compile的过程。

testGene = testGenerator("D:/matlabcode/stone/labelimage/images/1600",target_size = (256,256,3),num_image = 48,as_gray = False)
results = model.predict(testGene)
saveResultb("data/membrane/pre1600",results)