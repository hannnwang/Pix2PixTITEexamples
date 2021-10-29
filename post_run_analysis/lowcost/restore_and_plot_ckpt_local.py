#Apply a saved checkpoint to generate a few test instances. 
import tensorflow as tf
import os
import numpy as np
import glob
from matplotlib import pyplot as plt
import colorcet as cc
import math
import matplotlib.patches as mpatches

pi=math.pi

ckptname='ckpt9-60'

#Parameters to be edited. Should be checked if they are the same as in the training program (pix2pix_TF2_hw_train_scalar.py.)!
BATCH_SIZE = 1 #******This might change!
IMG_WIDTH = 256 #Size after random cropping
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 1 #Changed from 3 to 1
INPUT_CHANNELS = 1 #Added this, and changed from 3 to 1
NLAYERS = 5 #****This may change!

runname='lambda_'+'1000'+'_nlayers_5_'+'kitchen_sink'+'_700epochs_w_rotflipud/'
mpath='C:/Users/han/Desktop/Pix2Pix/scratch_outputs/'
PATH=mpath+runname

os.chmod(PATH, 0O777)

def similarity(fltar, flfak, method):
    """ Takes two SSH fields and compare them using given method """
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

#----------------Define Generator and Discriminator. The definitions should be exactly the same as in the training program (pix2pix_TF2_hw_train_scalar.py.)---------------------
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

#----------------------Defintion finished. Note that since we are not training, we don't need to define the loss functions.-----------------------------------------

# directory for saved tested images
testingpath = os.path.join(PATH, "test") 

#-------------------------------------Load the testing instances. ----------------------------------------

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
  bothimages=tf.data.Dataset.from_tensor_slices((input_image,real_image))
  return bothimages
#------------------------Loading finished-----------------------------------------



# -----------------------restoring the latest checkpoint in checkpoint_dir----------------------------------
checkpoint_dir = os.path.join(PATH, "training/training_checkpoints/")

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

ckptname=os.path.join(checkpoint_dir,ckptname)
checkpoint.restore(ckptname).assert_consumed

#-----------------------------------Run and plot test instances-------------------------------
#Running through all test data available under testingpath. If you don't want to run through all of them, you can change the next lines.
test_dataset_list=glob.glob(glob.escape(PATH)+'test/*.npy')
test_dataset = load_image_test(test_dataset_list)
test_dataset = test_dataset.batch(BATCH_SIZE)

ntest=len(os.listdir(PATH+'test/'))

ifig=0
for inp, tar in test_dataset:
    filename=test_dataset_list[ifig]
    ifig=ifig+1
    prediction = generator(inp, training=True)    
    SSH_raw=inp[0,:,:,0]
    SSH_tar=tar[0,:,:,0]
    SSH_fak=prediction[0,:,:,0]
    SSH_raw_plot=SSH_raw.numpy()
    SSH_tar_plot=SSH_tar.numpy()
    SSH_fak_plot=SSH_fak.numpy()
    vmax_temp=1
    fig2,axs2=plt.subplots(2,2)
   
    
    subplt=axs2[0,0].imshow(SSH_raw_plot,vmin=-vmax_temp,vmax=vmax_temp,cmap=cc.cm.CET_D1A)
    axs2[0,0].set_title('Input') #raw ssh
    axs2[0,0].set_axis_off()
    cbar = fig2.colorbar(subplt,ax=axs2[0,0],fraction=0.046, pad=0.04)
    
    subplt=axs2[0,1].imshow(SSH_tar_plot,vmin=-vmax_temp,vmax=vmax_temp,cmap=cc.cm.CET_D1A)
    axs2[0,1].set_title('Truth') #eta_cos^sim 
    axs2[0,1].set_axis_off()
    cbar = fig2.colorbar(subplt,ax=axs2[0,1],fraction=0.046, pad=0.04)
    
    subplt=axs2[1,0].imshow(SSH_fak_plot,vmin=-vmax_temp,vmax=vmax_temp,cmap=cc.cm.CET_D1A)
    axs2[1,0].set_title('Generated') #eta_cos^gen
    axs2[1,0].set_axis_off()
    cbar = fig2.colorbar(subplt,ax=axs2[1,0],fraction=0.046, pad=0.04)
    
    difference=SSH_tar_plot-SSH_fak_plot
    cor_diff=similarity(SSH_tar_plot, difference, "Correlation")#correlation between Truth and Generated
    subplt=axs2[1,1].imshow(difference,vmin=-vmax_temp,vmax=vmax_temp,cmap=cc.cm.CET_D1A)
    axs2[1,1].set_title('Difference, cor = %1.2f' %cor_diff)
    axs2[1,1].set_axis_off()
    cbar = fig2.colorbar(subplt,ax=axs2[1,1],fraction=0.046, pad=0.04)
    cbar.minorticks_on()

    cor_fig=similarity(SSH_tar_plot, SSH_fak_plot, "Correlation")
    L1_relative_fig=similarity(SSH_tar, SSH_fak, "L1_relative")  
    
    
    
    plt.suptitle(filename[-20:-4]+'\nCorrelation = %1.2f' %cor_fig + ', L1 = %1.2f' %L1_relative_fig) #Correlation between Truth and Difference
    plt.tight_layout()

