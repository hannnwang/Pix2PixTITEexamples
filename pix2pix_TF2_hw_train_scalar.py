#Restore checkpoints and run; if it's the first job, run from initial condition.
import tensorflow as tf

import os
import time

#from matplotlib import pyplot as plt

#from IPython import display

import argparse #added this

import psutil #Memory check

import gc #garbage collector for memory

import numpy as np

#import shutil  

import glob

#parser arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing training and testing images")  #=$SLURM_TMPDIR
parser.add_argument("--output_dir", required=True,
                    help="where to put output files") #/scratch/hannn/pix2pix-for-swot/blabla
parser.add_argument("--batch_size", type=int, default=1,
                    help="number of images in batch")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--LAMBDA", type=int, help="parameter LAMBDA")
parser.add_argument("--n_layers", type=int, help="number of layers in the discriminator. Default from master_scalar.bash: 3, which corresponds to 30-by-30 patches in the output")
a = parser.parse_args()
  
PATH = a.input_dir

BUFFER_SIZE = 3000 #3000 is way larger than the training set we have in all ES1-5. For other projects, this may need to be modified.
BATCH_SIZE = a.batch_size
IMG_WIDTH = 256 #Size after random cropping
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 1 #Changed from 3 to 1
INPUT_CHANNELS = 1 #changed from 3 to 1
NLAYERS=a.n_layers 
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

  # Downsampling
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
    
    if (NLAYERS == 3): #patchGAN
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
    elif (NLAYERS == 5): #imageGAN
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
        tf.print('ERROR: N_layers is not implemented!') 
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
    data=np.load(onefile) 
    image=data
    #Normalization is already done in SnapPrints. 
    
    image= tf.convert_to_tensor(image, dtype=tf.float32) #I don't know why this has to be done but it does. 

    w = tf.shape(image)[1]

    #Note: in facade data in the example code, the real (target) images are to the left of input image. Note that here the order is reversed!
    w = w // 2
    input_image = image[:, :w, :]  #Take left half; order reversed from the tensorflow tutorial code.
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

  bothimages=tf.data.Dataset.from_tensor_slices((input_image,real_image))#clumsy; there may be better ways to do this.
  return bothimages

test_dataset=glob.glob(PATH+'/test/*.npy')
test_dataset = load_image_test(test_dataset)
test_dataset = test_dataset.batch(BATCH_SIZE)

#Load the latest checkpoint and generate images

# restoring the latest checkpoint in checkpoint_dir
checkpoint_dir = os.path.join(a.output_dir, "training/training_checkpoints/") 
if not os.path.exists(checkpoint_dir):
   os.makedirs(checkpoint_dir)
   
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).assert_consumed #reports error if it's not restored. 

if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
    i_checkpoint=np.load((a.output_dir+"/i_checkpoint.npy"))
    mem_cost_list=np.load((a.output_dir+"/memory_epoch.npy"))
else:
    print("Initializing from scratch.")
    i_checkpoint=np.int_(1) #An optional integer, or an integer-dtype Variable or Tensor, used to number the checkpoint.
    mem_cost_list=np.array([0])


def random_crop(input_image, real_image):#***changed big time
   stacked_image = tf.stack([input_image, real_image], axis=0)
   cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, INPUT_CHANNELS])
   return cropped_image[0], cropped_image[1]


@tf.function() 
def random_jitter(input_image,real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)
    if  tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
          
    #Include random rotation and vertical flipping too:
    if tf.random.uniform(()) > 0.5:
        #flipud 
        input_image = tf.image.flip_up_down(input_image)
        real_image = tf.image.flip_up_down(real_image)              
    if tf.random.uniform(()) > 0.5:
        #rot90
        input_image = tf.image.rot90(input_image,k=1)
        real_image = tf.image.rot90(real_image,k=1)        
    
    return input_image, real_image


def change_image_train(input_image,real_image): #Changed indentation
  train_dataset = random_jitter(input_image,real_image) 
  return train_dataset

input_image_train, real_image_train = load(glob.glob(PATH+'/train/*.npy')) 
train_dataset = tf.data.Dataset.from_tensor_slices((input_image_train,real_image_train))
train_dataset = train_dataset.map(change_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)


train_dataset = train_dataset.shuffle(BUFFER_SIZE) #Note: in tf2, reshuffle_each_iteration=True
train_dataset = train_dataset.batch(BATCH_SIZE)

#Check if the shapes look right
print('shape of the first two pairs of "images" in the training data set:')
for f,g in train_dataset.take(2):
  print(f.shape)  
  print(g.shape)  
  
#Generator loss
LAMBDA = a.LAMBDA

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss
  

#Discriminator loss
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output) #loss from recognizing generated output 

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

EPOCHS = a.max_epochs

#import datetime
log_dir=os.path.join(a.output_dir, "training/logs/") 

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/")


checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt"+str(i_checkpoint))

#Make one prediction (this is not just pedagogical; if I delete it, somehow the graph would not be completely updated at each iteration, and there would be memory leak.)
for example_input, example_target in test_dataset.take(1):
   prediction = generator(example_input, training=True)


 
@tf.function
def train_step(input_image, target, epoch):
  gc.collect()#added this
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)

testingpath= os.path.join(a.output_dir, "testing/") 

def fit(train_ds, epochs, test_ds,mem_cost_list,i_checkpoint):   

     
  for epoch in range(epochs):
    start = time.time()
    """
    #**(debugging) save the train_ds at the start of epoch
    ids=0
    for inp,tar in train_ds:
      ids=ids+1
      with open(os.path.join(PATH,'','train_ds/train_ds_ep_%i' % epoch+'_%i.npz' % ids), 'wb') as f:
         np.savez(f, inp=inp,tar=tar)    
     """    
    print("Epoch: ", epoch)
    print('Memory used:') 
    mem=psutil.virtual_memory()
    print(mem.used)
    mem_cost_list=np.append(mem_cost_list,[mem.used]) #Not going to change the written file yet
    
    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0: 
        print()
      train_step(input_image, target, epoch)
    print()

    #saving (checkpoint) the model every few epochs
    if (epoch + 1) % 10 == 0: #Saving every 10 epochs; you can modify this if you want to save more/less frequently.
      checkpoint.save(file_prefix = checkpoint_prefix)
    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  i_checkpoint=i_checkpoint+1
  
  with open(a.output_dir+'/memory_epoch.npy','wb') as f: 
    np.save(f, mem_cost_list)
  with open(a.output_dir+'/i_checkpoint.npy','wb') as f: 
    np.save(f, i_checkpoint)



fit(train_dataset, EPOCHS, test_dataset,mem_cost_list,i_checkpoint)


