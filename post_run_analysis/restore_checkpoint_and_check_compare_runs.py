#Restore checkpoints. To run this, need to edit BATCH_SIZE and PATH.
import tensorflow as tf

import os

import numpy as np

import argparse 

import glob

from scipy import stats

#parser arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing training and testing images")  
parser.add_argument("--nlayers", type=int, help="nlayers in the discriminator")  

a = parser.parse_args()
PATH = a.input_dir
#PATH='/scratch/hannn/pix2pix-for-swot/cmssh-gray_cmcos-gray_testing-wp50_1000epochs/' #gray wp75 contains faketest 
#PATH='C:/Users/han/Desktop/Pix2Pix/scratchtemps/'

#Parameters to be edited. Should be checked if they are the same as in the training program!
BATCH_SIZE = 1 #******This changes very often!
IMG_WIDTH = 256 #Size after random cropping
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 1 #Changed from 3 to 1
INPUT_CHANNELS = 1 #Added this, and changed from 3 to 1
NLAYERS=a.nlayers #***This might change

os.chmod(PATH, 0O777)
print('Our current directory is:')
print(PATH)

#Generator:
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[256,256,INPUT_CHANNELS])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()

#Discriminator 
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inp = tf.keras.layers.Input(shape=[256, 256, INPUT_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, INPUT_CHANNELS], name='target_image')
    
    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)
    
    if (NLAYERS == 3):
        down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
        down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
        down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)
        
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1) # (bs, 31, 31, 512)
        
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
        
        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)
    elif (NLAYERS == 5): 
        down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
        down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
        down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)
        down4 = downsample(512, 4)(down3) # (bs, 16, 16, 512)
        down5 = downsample(512, 4)(down4) # (bs, 8, 8, 512)
        
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down5) # (bs, 10, 10, 512)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1) # (bs, 7, 7, 512)
        
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 9, 9, 512)
        
        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2) # (bs, 6, 6, 1)         
    else:
        tf.print('ERROR: N_layers is not implemented!') #Could also automate this by a for loop.
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()

# directory for saved tested images
testingpath = os.path.join(PATH, "testing") 

#Note: each .npz file was constructed so that the first variable is the ssh scalar. It has dimension width*(2 width)*1.
def load(image_file): 
  count=0
  input_list=tf.constant(0, dtype=tf.float32) 
  real_list=tf.constant(0, dtype=tf.float32)
  for onefile in image_file:
    #print('onefile is:')
    #print(onefile)
    data=np.load(onefile) 
    image=data
    #Normalization is already done in SnapPrints. To look for the normalization factor, look for the files in the project folder, created by create-scalar-files.bash (Calling SnapPrints_scalar)
    
    image= tf.convert_to_tensor(image, dtype=tf.float32) #hopefully this works

    ##Convert it to RGB, since the .npz file variable only contains one color channel.
    #image=tf.image.grayscale_to_rgb(image)#It basically just replicates the current channel to G and B channel. 
        
    w = tf.shape(image)[1]

    #Note: in facade data in the example code, the real (target) images come before (i.e. to the left of) input image. Note that in the official code the order is reversed.
    w = w // 2
    input_image = image[:, :w, :]  #Take left half; order reversed from official code
    real_image = image[:, w:, :] 

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    
    input_image=tf.expand_dims(input_image, axis=0) #for dataset packing. 
    real_image=tf.expand_dims(real_image, axis=0)
    
    if count==0:
        input_list=input_image
        real_list=real_image
    else:
        input_list=tf.concat([input_list,input_image],0)
        real_list=tf.concat([real_list,real_image],0)
    count=count+1
  return input_list, real_list

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image, real_image


def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
 # input_image, real_image = normalize(input_image, real_image) (normalization to [-1,1] is already done in SnapPrints_scalar called by create-scalar-files)
  bothimages=tf.data.Dataset.from_tensor_slices((input_image,real_image))#***changed big time, may be problematic
  return bothimages


test_dataset=glob.glob(glob.escape(PATH)+'/test/*.npy')#glob.escape is important if the PATH includes special characters like [ab].

#test_dataset = os.path.join(PATH,"test/",os.listdir(os.path.join(PATH, "test/")))
#print('test_dataset in list form:')
#print(test_dataset)
test_dataset = load_image_test(test_dataset)
#test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

ntest=len(os.listdir(PATH+'test/'))


#Load the latest checkpoint and generate images

# restoring the latest checkpoint in checkpoint_dir
checkpoint_dir = os.path.join(PATH, "training/training_checkpoints/")

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).assert_consumed
       
files=os.listdir(checkpoint_dir)
files = [os.path.join(checkpoint_dir, f) for f in files] # add path to each file
files.sort(key=lambda x: os.path.getmtime(x))

#Checking and plotting things

#Correlation between mapped ssh values, copied from Nico

def dst(a, b, distance):
    """ Computes the distance in RGB space between a series of RGB values and
    a given RGB value
    INPUT:
    a is a n-dimensional numpy array (n>1), last dimension is 3 (RGB)
    b is a len(3) list or array or tuple
    OUTPUT: (n-1)-dimensional array (distance between each element of the cmap
    """
    #print('shape of a:')
    #print(tf.shape(a))
    #print('shape of b:')
    #print(tf.shape(b))
    
    sa = a.shape  # assuming the last dimension is always 3 (for RGB)
    shape_pic = sa[:-1]

    # Flatten the array
    new_len = 1  # future total number of pixels
    for ss in shape_pic:
        new_len *= ss
   # aa = a.reshape(new_len, 3)  # we remove dimensions here
    aa=tf.reshape(a,[new_len, 3]) # we remove dimensions here  #HW: changed to tf.reshape
    #print('shape of aa:')
    #print(tf.shape(aa))
    if distance == 'L1':
        res = abs(aa[:, 0]-b[0]) + abs(aa[:, 1]-b[1]) + abs(aa[:, 2]-b[2])
    elif distance == 'L2':
        res = ((aa[:, 0]-b[0])**2 + (aa[:, 1]-b[1])** 2 + (aa[:, 2]-b[2])**2)**.5
    else:
        raise NameError("Distance " + distance + " not implemented yet.")
    #print('shape of res before reshaping')
    #print(tf.shape(res))
    #HW: changing it to tf.reshape
    vreturn=tf.reshape(res,shape_pic)
    return vreturn


def similarity(fltar, flfak, method):
    """ Takes two SSH fields are compare them using given method """
    if method == "Correlation":
        CorrMat = np.corrcoef(fltar.flat, flfak.flat)
        QF = CorrMat[0, 1]
    elif method == "L1":
        QF = np.average(abs(fltar-flfak))
    elif method == "L2":
        QF = np.average((fltar-flfak)**2)
    elif method == "L1_relative":
        QF = np.average(abs(fltar-flfak))/np.average(abs(fltar))
    elif method == "L2_relative":
        QF = np.average((fltar-flfak)**2)/np.average(fltar**2)
    else:
        raise NameError("Similarity method " + method + " not implemented")

    return QF  # quality factor

#ncdir = "/home/hannn/projects/def-ngrisoua/hannn/original-data" #Has to be in this weird format otherwise Graham does not recognize

#Construct filenames (by hw)
files=os.listdir(checkpoint_dir)

files = [item for item in files if item.endswith('.index')]

files = [os.path.join(checkpoint_dir, f) for f in files] # add path to each file
files.sort(key=lambda x: os.path.getmtime(x)) #Sort according to time
files =   [x[:-6] for x in files] #Now, only have files like "ckpt910-1004"

nckpt=len(files)

epochs_recorded=np.linspace(10,nckpt*10,nckpt)
cor_mean=np.empty([nckpt,OUTPUT_CHANNELS])
cor_std=np.empty([nckpt,OUTPUT_CHANNELS])
L1_relative_mean=np.empty([nckpt,OUTPUT_CHANNELS])
L1_relative_std=np.empty([nckpt,OUTPUT_CHANNELS])
L2_relative_mean=np.empty([nckpt,OUTPUT_CHANNELS])
L2_relative_std=np.empty([nckpt,OUTPUT_CHANNELS])
for ickpt in range(nckpt):
  ifile=ickpt
  filename=files[ifile]
  print('checkpoint name:')
  print(filename)
  checkpoint.restore(filename).assert_consumed
  cor_oneckpt=np.empty([ntest,OUTPUT_CHANNELS])
  L1_relative_oneckpt=np.empty([ntest,OUTPUT_CHANNELS])
  L2_relative_oneckpt=np.empty([ntest,OUTPUT_CHANNELS])
  ifig=0
  for inp, tar in test_dataset: #running through all test data
     prediction = generator(inp, training=True)  
     #inp=inp.numpy()
     tar=tar.numpy()
     prediction=prediction.numpy()
     for idim in range(OUTPUT_CHANNELS):
        #SSH_raw=inp[0,:,:,idim]
        SSH_fak=prediction[0,:,:,idim]
        SSH_tar=tar[0,:,:,idim]
        #with open(os.path.join(PATH+'test_checkpoint/testing_list_%i' % ifig +'_d%i.npz' %idim), 'wb') as f:
         #np.savez(f, SSH_raw=SSH_raw,SSH_fak=SSH_fak,SSH_tar=SSH_tar)    
        cor_oneckpt[ifig,idim]=similarity(SSH_tar, SSH_fak, "Correlation")
        L1_relative_oneckpt[ifig,idim]=similarity(SSH_tar, SSH_fak, "L1_relative")
        L2_relative_oneckpt[ifig,idim]=similarity(SSH_tar, SSH_fak, "L2_relative")

     ifig=ifig+1
  for idim in range(OUTPUT_CHANNELS):
     cor_mean[ickpt,idim] = np.mean(cor_oneckpt[:,idim])
     cor_std[ickpt,idim] = stats.sem(cor_oneckpt[:,idim])
     L1_relative_mean[ickpt,idim] = np.mean(L1_relative_oneckpt[:,idim])
     L1_relative_std[ickpt,idim] = stats.sem(L1_relative_oneckpt[:,idim])
     L2_relative_mean[ickpt,idim] = np.mean(L2_relative_oneckpt[:,idim])
     L2_relative_std[ickpt,idim] = stats.sem(L2_relative_oneckpt[:,idim])
                       
np.savez(PATH+'similarity_list_per_epoch',epochs_recorded=epochs_recorded,cor_mean=cor_mean,cor_std=cor_std,
           L1_relative_mean=L1_relative_mean,L1_relative_std=L1_relative_std,
           L2_relative_mean=L2_relative_mean,L2_relative_std=L2_relative_std)
