import os
from keras import models
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
    learning_rate = params['learning_rate']

    encoder_dataset = datasets['encoder']
    decoder_dataset = datasets['decoder']
    classifier_dataset = datasets.get('classifier')

    metrics = ['accuracy']
    optimizer = optimizers.RMSprop(lr=learning_rate, decay=decay)
    classifier_loss = 'categorical_crossentropy'

    # HACK: Keras Bug https://github.com/fchollet/keras/issues/5221
    # Sharing a BatchNormalization layer corrupts the graph
    # Workaround: carefully call _make_train_function

    def wgan_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    build_encoder = getattr(model_definitions, encoder_model)
    encoder = build_encoder(dataset=encoder_dataset, **params)

    build_decoder = getattr(model_definitions, decoder_model)
    decoder = build_decoder(dataset=decoder_dataset, **params)

    if enable_discriminator:
        build_discriminator = getattr(model_definitions, discriminator_model)
        discriminator = build_discriminator(is_discriminator=True, dataset=decoder_dataset, **params)
        discriminator.compile(loss=wgan_loss, optimizer=optimizer, metrics=metrics)
        discriminator._make_train_function()

        cgan = models.Model(inputs=decoder.inputs, outputs=discriminator(decoder.output))
        # The cgan model should train the generator, but not the discriminator
        cgan.layers[-1].trainable = False
        cgan.compile(loss=wgan_loss, optimizer=optimizer, metrics=metrics)
        cgan._make_train_function()
    else:
        discriminator = models.Sequential()
        cgan = models.Sequential()

    if enable_classifier:
        build_classifier = getattr(model_definitions, classifier_model)
        classifier = build_classifier(dataset=classifier_dataset, **params)

        transclassifier = models.Model(inputs=encoder.inputs, outputs=classifier(encoder.output))
        transclassifier.compile(loss=classifier_loss, optimizer=optimizer, metrics=metrics)
    else:
        classifier = models.Sequential()
        transclassifier = models.Sequential()

    if decoder_datatype == 'img':
        if enable_perceptual_loss:
            P = applications.mobilenet.MobileNet(include_top=False)
            perceptual_outputs = []
            for layer in P.layers:
                if layer.name.startswith('conv') and len(perceptual_outputs) < perceptual_layers:
                    perceptual_outputs.append(layer.output)
            print("Perceptual Loss: Using {} convolutional layers".format(len(perceptual_outputs)))

            texture = models.Model(inputs=P.inputs, outputs=perceptual_outputs)

            # A scalar value that measures the perceptual difference between two images wrt. a pretrained convnet
            def perceptual_loss(y_true, y_pred):
                T_a, T_b = texture(y_true), texture(y_pred)
                p_loss = K.mean(K.abs(T_a[0] - T_b[0]))
                for a, b in zip(T_a, T_b)[1:]:
                    G_a = K.mean(K.mean(a, axis=1), axis=1)
                    G_b = K.mean(K.mean(b, axis=1), axis=1)
                    p_loss += K.mean(K.abs(G_a - G_b))
                return p_loss

            transcoder_loss = lambda x, y: alpha * losses.mean_absolute_error(x, y) + (1 - alpha) * perceptual_loss(x, y)
        else:
            transcoder_loss = losses.mean_absolute_error
    else:
        transcoder_loss = losses.categorical_crossentropy

    transcoder = models.Model(inputs=encoder.inputs, outputs=decoder(encoder.output))
    transcoder.compile(loss=transcoder_loss, optimizer=optimizer, metrics=metrics)
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
        'cgan': cgan, 
        'classifier': classifier,
        'transclassifier': transclassifier,
    }


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
