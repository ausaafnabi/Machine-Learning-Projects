def build_encoder(inputs , num_labels,feature1_dim=256):
    """ Build the Classifier (Encoder) Model sub networks
    Two sub networks:
    1) Encoder0: Image to feature1 (intermediate latent feature)
    2) Encoder1: feature1 to labels
    
    # Arguments
        inputs (Layers): x - images, feature1 - feature1 layer output
        num_labels (int): number of class labels
        feature1_dim (int): feature1 dimensionality
    # Returns
        enc0, enc1 (Models): Description below
    """
    kernel_size=3
    filters=64

    x,feature1=inputs
    #Encoder0 or enc0
    y = Conv2D(filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu')(x)
    y = MaxPooling2D()(y)
    y = Conv2D(filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu')(y)
    y = MaxPooling2D()(y)
    y=Flatten()(y)
    feature1_output = Dense(feature1_dim,activation='relu')(y)
    #Encoder0 or enc0:image to feature1
    enc0 = Model(inputs=x,outputs=feature1_output,name="encoder0")

    #Encoder1 or enc1
    y= Dense(num_labels)(feature1)
    labels = Activation('softmax')(y)
    # Encoder1 or enc1 : feature1 to class_labels
    enc1 = Model(inputs=feature1,outputs= labels,name="encoder1")

    # return both enc0 and enc1
    return enc0 ,enc1

def build_generator(latent_codes,image_size,feature1_dim=256):
    """Build Generator Model sub networks
    Two sub networks:
    1) Class and noise to feature1 (intermediate feature)
    2) feature1 to image

    # Arguments
        latent_codes (Layers): discrete code (labels), noise and feature1 features
        image_size (int): Target size of one side (assuming square image)
        feature1_dim (int): feature1 dimensionality
    # Returns
        gen0, gen1 (Models): Description below
    """

    #Latent codes and network parameters
    labels,z0,z1,feature1= latent_codes
    #image_resize= image_size//4
    #kernel_size=5
    #layer_filters =[128,64,32,1]

    #gen1 inputs
    inputs = [labels,z1]    #10+50 = 63-dim
    x = concatenate(inputs,axis=1)
    x = Dense(512,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512,activation='relu')(x)
    x = BatchNormalization()(x)
    fake_feature1 = Dense(feature1_dim,activation='relu')(x)
    #gen1 :classes and noise(feature2+z1) to feature1
    gen1 = Model(inputs,fake_feature1,name='gen1')

    #gen0: feature1 +z0 to feature0 (image)
    gen0 = gan.generator(feature1,image_size,codes=z0)
    
    return gen0,gen1

def build_discriminator(inputs,z_dim=50):


    #inputs  is 256-dim feature1
    x= Dense(256,activation='relu')(inputs)
    x = Dense(256,activation='relu')(x)

    #first output is probability that feature1 is real
    f1_source = Dense(1)(x)
    f1_source = Activation('sigmoid',name='feature1_source')(f1_source)
    #z1 reconstruction (Q1 network)

    z1_recon =Dense(z_dim)(x)
    z1_recon = Activation('tanh' , name='z1')(z1_recon)

    discriminator_outputs = [f1_source,z1_recon]
    dis1 = Model(inputs,discriminator_outputs,name='dis1')
    return dis1 


def build_and_train_models():
    # load MNIST dataset
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
    # reshape and normalize images
	image_size = x_train.shape[1]
	x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
	x_train = x_train.astype('float32') / 255	
	x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
	x_test = x_test.astype('float32') / 255
	# number of labels
	num_labels = len(np.unique(y_train))
	# to one-hot vector
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	model_name = "stackedgan_mnist"
	# network parameters
	batch_size = 64	
	train_steps = 40000	
	lr = 2e-4
	decay = 6e-8
	input_shape = (image_size, image_size, 1)
	label_shape = (num_labels, )
	z_dim = 50
	z_shape = (z_dim, )
	feature1_dim = 256
	feature1_shape = (feature1_dim, )
	# build discriminator 0 and Q network 0 models
	inputs = Input(shape=input_shape, name='discriminator0_input')
	dis0 = gan.discriminator(inputs, num_codes=z_dim)
	# [1] uses Adam, but discriminator converges easily with RMSprop
	optimizer = RMSprop(lr=lr, decay=decay)
	# loss fuctions: 1) probability image is real (adversarial0 loss)
	# 2) MSE z0 recon loss (Q0 network loss or entropy0 loss)
	loss = ['binary_crossentropy', 'mse']
	loss_weights = [1.0, 10.0]	
	dis0.compile(loss=loss,
	loss_weights=loss_weights,
	optimizer=optimizer,
	metrics=['accuracy'])
	dis0.summary() # image discriminator, z0 estimator
	# build discriminator 1 and Q network 1 models
	input_shape = (feature1_dim, )
	inputs = Input(shape=input_shape, name='discriminator1_input')
	dis1 = build_discriminator(inputs, z_dim=z_dim )
	# loss fuctions: 1) probability feature1 is real (adversarial1 loss)
	# 2) MSE z1 recon loss (Q1 network loss or entropy1 loss)
	loss = ['binary_crossentropy', 'mse']
	loss_weights = [1.0, 1.0]
	dis1.compile(loss=loss,
	loss_weights=loss_weights,
	optimizer=optimizer,
	metrics=['accuracy'])
	dis1.summary() # feature1 discriminator, z1 estimator
	# build generator models
	feature1 = Input(shape=feature1_shape, name='feature1_input')
	labels = Input(shape=label_shape, name='labels')
	z1 = Input(shape=z_shape, name="z1_input")
	z0 = Input(shape=z_shape, name="z0_input")
	latent_codes = (labels, z0, z1, feature1)
	gen0, gen1 = build_generator(latent_codes, image_size)
	gen0.summary() # image generator
	gen1.summary() # feature1 generator
	# build encoder models
	input_shape = (image_size, image_size, 1)
	inputs = Input(shape=input_shape, name='encoder_input')
	enc0, enc1 = build_encoder((inputs, feature1), num_labels)
	enc0.summary() # image to feature1 encoder
	enc1.summary() # feature1 to labels encoder (classifier)
	encoder = Model(inputs, enc1(enc0(inputs)))
	encoder.summary() # image to labels encoder (classifier)
	data = (x_train, y_train), (x_test, y_test)
	train_encoder(encoder, data, model_name=model_name)
	# build adversarial0 model =
	# generator0 + discriminator0 + encoder0
	optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
	# encoder0 weights frozen
	enc0.trainable = False
	# discriminator0 weights frozen
	dis0.trainable = False
	gen0_inputs = [feature1, z0]
	gen0_outputs = gen0(gen0_inputs)
	adv0_outputs = dis0(gen0_outputs) + [enc0(gen0_outputs)]
	# feature1 + z0 to prob feature1 is
	# real + z0 recon + feature0/image recon
	adv0 = Model(gen0_inputs, adv0_outputs, name="adv0")
	# loss functions: 1) prob feature1 is real (adversarial0 loss)
	# 2) Q network 0 loss (entropy0 loss)
	# 3) conditional0 loss
	loss = ['binary_crossentropy', 'mse', 'mse']
	loss_weights = [1.0, 10.0, 1.0]
	adv0.compile(loss=loss,
				loss_weights=loss_weights,	
				optimizer=optimizer,
				metrics=['accuracy'])
	adv0.summary()
	# build adversarial1 model =
	# generator1 + discriminator1 + encoder1
	# encoder1 weights frozen
	enc1.trainable = False
	# discriminator1 weights frozen
	dis1.trainable = False
	gen1_inputs = [labels, z1]
	gen1_outputs = gen1(gen1_inputs)
	adv1_outputs = dis1(gen1_outputs) + [enc1(gen1_outputs)]
	# labels + z1 to prob labels are real + z1 recon + feature1 recon
	adv1 = Model(gen1_inputs, adv1_outputs, name="adv1")
	# loss functions: 1) prob labels are real (adversarial1 loss)
	# 2) Q network 1 loss (entropy1 loss)
	# 3) conditional1 loss (classifier error)
	loss_weights = [1.0, 1.0, 1.0]
	loss = ['binary_crossentropy', 'mse', 'categorical_crossentropy']
	adv1.compile(loss=loss,
	loss_weights=loss_weights,
	optimizer=optimizer,
	metrics=['accuracy'])
	adv1.summary()
	# train discriminator and adversarial networks
	models = (enc0, enc1, gen0, gen1, dis0, dis1, adv0, adv1)
	params = (batch_size, train_steps, num_labels, z_dim, model_name)
	train(models, data, params)


def train(models, data, params):
	"""Train the discriminator and adversarial Networks
	Alternately train discriminator and adversarial networks by batch.
	Discriminator is trained first with real and fake images,
	corresponding one-hot labels and latent codes.
	Adversarial is trained next with fake images pretending
	to be real,
	corresponding one-hot labels and latent codes.
	Generate sample images per save_interval.
	# Arguments
		models (Models): Encoder, Generator, Discriminator,Adversarial models
		data (tuple): x_train, y_train data
		params (tuple): Network parameters
	"""
	# the StackedGAN and Encoder models
	enc0, enc1, gen0, gen1, dis0, dis1, adv0, adv1 = models
	# network parameters
	batch_size, train_steps, num_labels, z_dim, model_name = params
	# train dataset
	(x_train, y_train), (_, _) = data
	# the generator image is saved every 500 steps
	save_interval = 500
	# label and noise codes for generator testing
	z0 = np.random.normal(scale=0.5, size=[16, z_dim])
	z1 = np.random.normal(scale=0.5, size=[16, z_dim])
	noise_class = np.eye(num_labels)[np.arange(0, 16) % num_labels]
	noise_params = [noise_class, z0, z1]
	# number of elements in train dataset
	train_size = x_train.shape[0]
	print(model_name,
			"Labels for generated images: ",
			np.argmax(noise_class, axis=1))
	for i in range(train_steps):
		# train the discriminator1 for 1 batch
		# 1 batch of real (label=1.0) and fake feature1 (label=0.0)
		# randomly pick real images from dataset
		rand_indexes = np.random.randint(0, train_size, size=batch_size)
		real_images = x_train[rand_indexes]
		# real feature1 from encoder0 output
		real_feature1 = enc0.predict(real_images)
		# generate random 50-dim z1 latent code
		real_z1 = np.random.normal(scale=0.5, size=[batch_size,z_dim])
		# real labels from dataset
		real_labels = y_train[rand_indexes]
		# generate fake feature1 using generator1 from
		# real labels and 50-dim z1 latent code
		fake_z1 = np.random.normal(scale=0.5, size=[batch_size,z_dim])
		fake_feature1 = gen1.predict([real_labels, fake_z1])
		# real + fake data
		feature1 = np.concatenate((real_feature1, fake_feature1))
		z1 = np.concatenate((fake_z1, fake_z1))
		# label 1st half as real and 2nd half as fake
		y = np.ones([2 * batch_size, 1])
		y[batch_size:, :] = 0
		# train discriminator1 to classify feature1
		# as real/fake and recover
		# latent code (z1). real = from encoder1,
		# fake = from genenerator1
		# joint training using discriminator part of advserial1 loss
		# and entropy1 loss
		metrics = dis1.train_on_batch(feature1, [y, z1])
		# log the overall loss only (fr dis1.metrics_names)
		log = "%d: [dis1_loss: %f]" % (i, metrics[0])
		# train the discriminator0 for 1 batch
		# 1 batch of real (label=1.0) and fake images (label=0.0)
		# generate random 50-dim z0 latent code
		fake_z0 = np.random.normal(scale=0.5, size=[batch_size,z_dim])
		# generate fake images from real feature1 and fake z0
		fake_images = gen0.predict([real_feature1, fake_z0])
		# real + fake data
		x = np.concatenate((real_images, fake_images))
		z0 = np.concatenate((fake_z0, fake_z0))
		# train discriminator0 to classify image as real/fake and recover
		# latent code (z0)
		# joint training using discriminator part of advserial0 loss
		# and entropy0 loss
		metrics = dis0.train_on_batch(x, [y, z0])
		# log the overall loss only (fr dis0.metrics_names)
		log = "%s [dis0_loss: %f]" % (log, metrics[0])
		# adversarial training
		# generate fake z1, labels
		fake_z1 = np.random.normal(scale=0.5, size=[batch_size,z_dim])
		# input to generator1 is sampling fr real labels and
		# 50-dim z1 latent code
		gen1_inputs = [real_labels, fake_z1]
		# label fake feature1 as real
		y = np.ones([batch_size, 1])
		# train generator1 (thru adversarial) by
		# fooling the discriminator
		# and approximating encoder1 feature1 generator
		# joint training: adversarial1, entropy1, conditional1
		metrics = adv1.train_on_batch(gen1_inputs, [y, fake_z1,real_labels])
		fmt = "%s [adv1_loss: %f, enc1_acc: %f]"
		# log the overall loss and classification accuracy
		log = fmt % (log, metrics[0], metrics[6])
		# input to generator0 is real feature1 and
		# 50-dim z0 latent code
		fake_z0 = np.random.normal(scale=0.5, size=[batch_size,z_dim])
		gen0_inputs = [real_feature1, fake_z0]
		# train generator0 (thru adversarial) by
		# fooling the discriminator
		# and approimating encoder1 image source generator
		# joint training: adversarial0, entropy0, conditional0
		metrics = adv0.train_on_batch(gen0_inputs, [y, fake_z0, real_feature1])	
		# log the overall loss only
		log = "%s [adv0_loss: %f]" % (log, metrics[0])
		print(log)

		if (i + 1) % save_interval == 0:
			if (i + 1) == train_steps:
				show = True
			else:
				show = False
				generators = (gen0, gen1)
				plot_images(generators,
							noise_params=noise_params,
							show=show,
							step=(i + 1),
							model_name=model_name)

	# save the modelis after training generator0 & 1
	# the trained generator can be reloaded for
	# future MNIST digit generation
	gen1.save(model_name + "-gen1.h5")
	gen0.save(model_name + "-gen0.h5")
