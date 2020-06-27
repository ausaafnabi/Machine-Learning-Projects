

def generator(input, image_size):

    image_resize = image_size //4
    kernel_size = 5
    layer_filters = [128,64,32,1]
    
    x = Dense(image_resize*image_resize*layer_filters[0])(inputs)
    x = Reshape((image_resize,image_resize,layer_filter[0]))(x)
    for filters in layer_filters:
        if filters >layer_filters[-3]:
            strides =2
        else:
            strides =1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,kernel_size=kernel_size, strides=strides,padding='same')(x)
    x = Activation('sigmoid')(x)
    generator = Model(inputs,x,name='generator')
    return generator


def descriminator(inputs):

    kernel_size =5
    layer_filters=[32,64,128,256]

    x =inputs
    for filters in layer_filters:
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides =2

        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters, kernel_size=kernel_size,strides=strides,padding='same')(x)
    x = flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model(inputs,x,name='discriminator')
    return discriminator

def Build_and_train_models():
    (x_train,_),(_,_) = mnist.load_data()
    image_size = x_train.shape[1]
    x_train = x_train.astype('float32') /255
    model_name = "dcgan_mnist"

    latent_size =100
    batch_size =64
    train_steps= 40000
    lr = 2e-4
    decay=6e-8
    input_shape=(image_size,image_size,1)
    inputs = Input(shape=input_shape,name = 'discriminator_input')
    discriminator = discriminator(inputs)
    optimizer = RMSprop(lr = lr*0.5,decay=decay *0.5)
    discriminator.trainable=False
    adversarial - Model(inputs,discriminator(generator(inputs)),name = model_name)
    adversarial.compile(loss= 'binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    adversarial.summary()

    models = (generator,discriminator,adversarial)
    params=(batch_size,latent_size,train_steps,model_name)
    train(models,x_train,params)


def train(models, x_train, params):
    """Train the Discriminator and Adversarial Networks
    Alternately train Discriminaor and Adversarial networks by batch.
    Discriminator is trained first with properly real and fake images.
    Adversarial is trained next with fake images pretending to be real
    Generate sample images per save_interval.

    # Arguments
        models (list): Generator, Discriminator, Adversarial models
        x_train (tensor): Train images
        params (list) : Networks parameters
    """
    # the GAN models
    # generator, discriminator, adversarial = models
    # network parameters
    batch_size, latent_size, train_steps, model_name = params
    # the generator image is saved every 500 steps
    save_interval = 500
    # noise vector to see how the generator output evolves
    # during training
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    # number of elements in train dataset
    train_size = x_train.shape[0]
    for i in range(train_steps):
# train the discriminator for 1 batch
# 1 batch of real (label=1.0) and fake images (label=0.0)
# randomly pick real images from dataset
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[rand_indexes]
        # generate fake images from noise using generator
        # generate noise using uniform distribution
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        # generate fake images
        fake_images = generator.predict(noise)
        # real + fake images = 1 batch of train data
        x = np.concatenate((real_images, fake_images))
        # label real and fake images
        # real images label is 1.0
        y = np.ones([2 * batch_size, 1])
        # fake images label is 0.0
        y[batch_size:, :] = 0.0
        # train discriminator network, log the loss and accuracy
        loss, acc = discriminator.train_on_batch(x, y)
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)
        # train the adversarial network for 1 batch
        # only the generator is trained
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        y = np.ones([batch_size, 1])
# note that unlike in discriminator training,
# the fake images go to the discriminator input of the adversarial
# for classification
# log the loss and accuracy
        loss, acc = adversarial.train_on_batch(noise, y)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        if (i + 1) % save_interval == 0:
        if (i + 1) == train_steps:
            show = True
        else:
            show = False

            plot_images(generator, noise_input=noise_input,
                    show=show,
                    step=(i + 1),
                    model_name=model_name)
    generator.save(model_name + ".h5")
