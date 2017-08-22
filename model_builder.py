import os
from keras import models
from keras import layers
from keras import optimizers
from keras import applications
from keras import losses
from keras import backend as K

import model_definitions


def build_models(datasets, **params):
    encoder_model = params['encoder_model']
    decoder_model = params['decoder_model']
    discriminator_model = params['discriminator_model']
    classifier_model = params['classifier_model']
    enable_discriminator = params['enable_discriminator']
    enable_classifier = params['enable_classifier']
    enable_perceptual_loss = params['enable_perceptual_loss']
    alpha = params['perceptual_loss_alpha']
    perceptual_layers = params['perceptual_loss_layers']
    decay = params['decay']
    decoder_datatype = params['decoder_datatype']
    batch_size = params['batch_size']
    gan_type = params['gan_type']

    learning_rate = params['learning_rate']
    learning_rate_classifier = params['learning_rate_classifier']

    encoder_dataset = datasets['encoder']
    decoder_dataset = datasets['decoder']
    classifier_dataset = datasets.get('classifier')

    metrics = ['accuracy']
    classifier_optimizer = optimizers.Adam(lr=learning_rate * learning_rate_classifier, decay=decay, clipnorm=.1)
    autoenc_decay = decay * 2
    autoenc_optimizer = optimizers.Adam(lr=learning_rate, decay=autoenc_decay, clipnorm=.1)
    classifier_loss = 'categorical_crossentropy'

    # HACK: Keras Bug https://github.com/fchollet/keras/issues/5221
    # Sharing a BatchNormalization layer corrupts the graph
    # Workaround: carefully call _make_train_function

    build_encoder = getattr(model_definitions, encoder_model)
    encoder = build_encoder(dataset=encoder_dataset, **params)

    build_decoder = getattr(model_definitions, decoder_model)
    decoder = build_decoder(dataset=decoder_dataset, **params)

    if enable_discriminator:
        if gan_type == 'wgan-gp':
            build_discriminator = construct_wgan_gp
        elif gan_type == 'began':
            build_discriminator = construct_began
        discriminator, generator_discriminator = build_discriminator(decoder, datasets, **params)
    else:
        # Placeholder models for summary()
        discriminator = models.Sequential()
        generator_discriminator = models.Sequential()

    if enable_classifier:
        build_classifier = getattr(model_definitions, classifier_model)
        classifier = build_classifier(dataset=classifier_dataset, **params)

        transclassifier = models.Model(inputs=encoder.inputs, outputs=classifier(encoder.output))
        transclassifier.compile(loss=classifier_loss, optimizer=classifier_optimizer, metrics=metrics)
    else:
        classifier = models.Sequential()
        transclassifier = models.Sequential()

    if decoder_datatype == 'img':
        if enable_perceptual_loss and perceptual_layers > 0:
            P = applications.inception_v3.InceptionV3(include_top=False)
            perceptual_outputs = []
            for layer in P.layers:
                if 'conv' in layer.name and len(perceptual_outputs) < perceptual_layers:
                    perceptual_outputs.append(layer.output)
            print("Perceptual Loss: Using {} convolutional layers".format(len(perceptual_outputs)))

            texture = models.Model(inputs=P.inputs, outputs=perceptual_outputs)

            # Weigh loss function based on pixel location, to emphasize more important pixels
            # Penalizes pixel error near the center of the image, ignores the edges and corners
            def center_weighted_mean(x):
                import tensorflow as tf
                # Input dims are: batch x height x width x channels
                # Reduce the batch and the channels
                x = K.mean(x, axis=-1)
                x = K.mean(x, axis=0)

                # Construct a 1D function that peaks in the middle
                height = tf.shape(x)[1]
                scale = tf.cast(height, tf.float32) / 2.
                # HACK: add 1
                triangle = tf.minimum(tf.range(0, height), tf.range(height, 0, -1)) + 1
                linear_weight = tf.cast(triangle, tf.float32) / scale

                # Outer product of the function with itself to form a 2D mask
                weight_vertical = tf.expand_dims(linear_weight, axis=0)
                weight_horiz = tf.expand_dims(linear_weight, axis=1)
                mask = tf.matmul(weight_horiz, weight_vertical)

                # Reduce depth to 2D and pointwise multiply by the mask
                masked = tf.multiply(mask, x)

                # Reduce width/height for the final score
                return K.mean(K.mean(masked))

            # A scalar value that measures the perceptual difference between two images wrt. a pretrained convnet
            def perceptual_loss(y_true, y_pred):
                # Input dims are: batch x height x width x channels
                p_loss = (1 - alpha) * center_weighted_mean(K.square(y_true - y_pred))
                T_a, T_b = texture(y_true), texture(y_pred)
                for a, b in zip(T_a, T_b):
                    layer_weight = alpha / len(T_a)
                    p_loss += layer_weight * center_weighted_mean(K.square(a - b))
                return p_loss

            transcoder_loss = perceptual_loss
        else:
            transcoder_loss = losses.mean_absolute_error
    else:
        transcoder_loss = losses.categorical_crossentropy

    transcoder = models.Model(inputs=encoder.inputs, outputs=decoder(encoder.output))
    transcoder.compile(loss=transcoder_loss, optimizer=autoenc_optimizer, metrics=metrics)
    transcoder._make_train_function()

    print("\nEncoder")
    encoder.summary()
    print("\nDecoder")
    decoder.summary()
    print("\nDiscriminator")
    discriminator.summary()
    print('\nClassifier:')
    classifier.summary()

    return {
        'encoder': encoder,
        'decoder': decoder,
        'transcoder': transcoder,
        'discriminator': discriminator,
        'generator_discriminator': generator_discriminator,
        'classifier': classifier,
        'transclassifier': transclassifier,
    }


# Boundary Equilibrium GAN
# The discriminator is, itself, another autoencoder
# It minimizes reconstruction loss on real examples,
# but maximizes reconstruction loss on fake examples
# https://arxiv.org/pdf/1703.10717.pdf
def construct_began(decoder, datasets, **params):
    decoder_model = params['decoder_model']
    encoder_model = params['encoder_model']
    batch_size = params['batch_size']
    latent_size = params['thought_vector_size']
    learning_rate = params['learning_rate']
    decay = params['decay']

    encoder_dataset = datasets['encoder']
    decoder_dataset = datasets['decoder']

    began_optimizer = optimizers.Adam(lr=learning_rate, decay=decay)

    build_encoder = getattr(model_definitions, encoder_model)
    disc_encoder = build_encoder(dataset=encoder_dataset, **params)

    build_decoder = getattr(model_definitions, decoder_model)
    disc_decoder = build_decoder(dataset=decoder_dataset, **params)
    
    disc_input_real = layers.Input(batch_shape=disc_encoder.input.shape)
    disc_input_fake = layers.Input(batch_shape=disc_encoder.input.shape)

    disc_output_real = disc_decoder(disc_encoder(disc_input_real))
    disc_output_fake = disc_decoder(disc_encoder(disc_input_fake))

    def real_loss(y_true, y_pred):
        return K.mean(K.abs(y_pred - y_true))

    def fake_loss(y_true, y_pred):
        return 1 - K.mean(K.abs(y_pred - y_true))

    discriminator = models.Model(
            inputs=[disc_input_real, disc_input_fake],
            outputs=[disc_output_real, disc_output_fake])
    discriminator.compile(optimizer=began_optimizer, metrics=['accuracy'],
            loss=[real_loss, fake_loss])
    discriminator._make_train_function()

    # The generator_discriminator optimizes D(G(x; A); B) updating A and keeping B fixed
    latent_input = layers.Input(batch_shape=(batch_size, latent_size))
    gen_disc_output = disc_decoder(disc_encoder(decoder(latent_input)))
    generator_discriminator = models.Model(inputs=[latent_input], outputs=[gen_disc_output])
    generator_discriminator.layers[-1].trainable = False
    generator_discriminator.compile(loss=real_loss, optimizer=began_optimizer, metrics=['accuracy'])
    generator_discriminator._make_train_function()

    return discriminator, generator_discriminator


def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


def negative_l1_loss(y_true, y_pred):
    return 1 - K.mean(K.abs(y_true - y_pred))


# Wasserstein GAN with Gradient Penalty
# A WGAN discriminator is "an art critic, not a counterfitting expert"
# Discriminator is a CNN that outputs a scalar from -1 (fake) to 1 (realistic)
# The GP loss enforces Lipschitz continuity (prevents D from being too spiky or non-linear)
# In practice, GP loss has an effect similar to weight regularization
# https://arxiv.org/pdf/1704.00028.pdf
def construct_wgan_gp(decoder, datasets, **params):
    batch_size = params['batch_size']
    discriminator_model = params['discriminator_model']
    learning_rate = params['learning_rate']
    decay = params['decay']
    learning_rate_discriminator = params['learning_rate_disc']
    learning_rate_generator = params['learning_rate_generator']

    decoder_dataset = datasets['decoder']
    build_discriminator = getattr(model_definitions, discriminator_model)

    # Discriminator beta from Gulrajani et al
    disc_optimizer = optimizers.Adam(beta_1=0.5, beta_2=0.9, lr=learning_rate * learning_rate_discriminator, decay=decay, clipnorm=0.1)
    gen_optimizer = optimizers.Adam(beta_1=0.5, beta_2=0.9, lr=learning_rate * learning_rate_generator, decay=decay, clipnorm=.1)

    # This is the inner discriminator; we apply it to real, fake, and interpolated items
    discriminator_inner = build_discriminator(is_discriminator=True, dataset=decoder_dataset, **params)
    rand_interpolater = iwgan_interpolation_layer(batch_size)

    disc_input_real = layers.Input(batch_shape=discriminator_inner.input.shape)
    disc_input_fake = layers.Input(batch_shape=discriminator_inner.input.shape)
    interpolated = rand_interpolater([disc_input_real, disc_input_fake])

    disc_output_real = discriminator_inner(disc_input_real)
    disc_output_fake = discriminator_inner(disc_input_fake)
    # TODO: Should we try interpolating in latent space?
    disc_output_interpolated = discriminator_inner(interpolated)

    def gradient_penalty_loss(y_true, y_pred):
        return gradient_penalty(y_true, y_pred, wrt=interpolated)

    # The discriminator optimizes D(real), D(fake), and the gradient of D(interpolated)
    discriminator = models.Model(
            inputs=[disc_input_real, disc_input_fake],
            outputs=[disc_output_real, disc_output_fake, disc_output_interpolated])
    discriminator.compile(optimizer=disc_optimizer, metrics=['accuracy'],
            loss=[wasserstein_loss, wasserstein_loss, gradient_penalty_loss])
    discriminator._make_train_function()

    # The generator_discriminator optimizes D(G(x; A); B) updating A and keeping B fixed
    generator_discriminator = models.Model(inputs=decoder.inputs, outputs=discriminator_inner(decoder.output))
    # TODO: Make sure that layer -1 is always the discriminator
    generator_discriminator.layers[-1].trainable = False
    generator_discriminator.compile(loss=wasserstein_loss, optimizer=gen_optimizer, metrics=['accuracy'])
    generator_discriminator._make_train_function()
    return discriminator, generator_discriminator


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty(y_true, y_pred, wrt):
    # Hack: Ignores y_true, just computes a gradient magnitude
    gradients = K.gradients(K.sum(y_pred), wrt)
    gradient_l2_norm = K.sqrt(K.sum(K.square(gradients)))
    return K.square(1 - gradient_l2_norm)


# A Keras layer that outputs a uniform random linear interpolation of two inputs
def iwgan_interpolation_layer(batch_size):
    from keras.layers.merge import _Merge
    class RandomWeightedAverage(_Merge):
        def _merge_function(self, inputs):
            weights = K.random_uniform((batch_size, 1, 1, 1))
            return (weights * inputs[0]) + ((1 - weights) * inputs[1])
    return RandomWeightedAverage()



def load_weights(models, **params):
    encoder_weights = params['encoder_weights']
    decoder_weights = params['decoder_weights']
    discriminator_weights = params['discriminator_weights']
    classifier_weights = params['classifier_weights']
    enable_discriminator = params['enable_discriminator']
    enable_classifier = params['enable_classifier']

    if os.path.exists(encoder_weights):
        models['encoder'].load_weights(encoder_weights)
    if os.path.exists(decoder_weights):
        models['decoder'].load_weights(decoder_weights)
    if enable_discriminator and os.path.exists(discriminator_weights):
        models['discriminator'].load_weights(discriminator_weights)
    if enable_classifier and os.path.exists(classifier_weights):
        models['classifier'].load_weights(classifier_weights)
