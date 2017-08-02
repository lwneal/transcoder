import time
import numpy as np
from keras import backend as K
import imutil

def compute_trajectory(
        encoder, decoder, classifier,
        Z, classifier_dataset,
        **params):
    thought_vector_size = params['thought_vector_size']
    video_filename = params['video_filename']

    trajectory = []

    # Randomly choose a class to mutate toward
    selected_class = np.random.randint(0, len(classifier_dataset.idx_to_name))

    classification = classifier.predict(Z)[0]
    print("Starting class: {}  Counterfactual target class: {}".format(
        np.argmax(classification), selected_class))
    original_class = np.copy(classification)

    # This will contain our latent vector
    latent_value = K.placeholder((1, thought_vector_size))

    loss = K.variable(0.)
    loss -= K.log(classifier.outputs[0][0][selected_class])

    grads = K.gradients(loss, classifier.inputs[0])

    compute_gradient = K.function(classifier.inputs, grads)

    # Perform gradient descent on the classification loss
    # Save each point in the latent space trajectory
    Z = np.copy(Z)  # Hack to avoid unexpected surprise
    step_size = .01
    classification = classifier.predict(Z)[0]
    momentum = None
    NUM_FRAMES = 240

    for _ in range(10):
        trajectory.append(np.copy(Z))
    for i in range(10 * NUM_FRAMES):
        # Hack: take 10x steps until gradient gets large enough
        for _ in range(10):
            gradient = compute_gradient([Z])[0]
            if momentum is None:
                momentum = gradient
            momentum += gradient
            momentum *= .99
            # Clip magnitude to 1.0
            if np.linalg.norm(momentum) > 1.0:
                momentum /= np.linalg.norm(momentum)
                continue
        Z -= momentum * step_size
        # L2 regularization? may be unnecessary after gradient clipping
        #Z *= 0.99
        # Inspired by Michael Jordan's talk on saddle points
        # https://arxiv.org/pdf/1703.00887.pdf
        Z += np.random.normal(scale=.001, size=Z.shape)

        classification = classifier.predict(Z)[0]

        if i % 10 == 0:
            trajectory.append(np.copy(Z))

        # Early stop if we've completed the transition
        if K.get_session().run(loss, {classifier.inputs[0]: Z}) < .01:
            print("Counterfactual optimization ending after {} iterations".format(i))
            break

    for _ in range(5):
        trajectory.append(np.copy(Z))

    print('\n')
    print("Original Image:")
    img = decoder.predict(Z)[0]
    imutil.show(img, filename='{}_counterfactual_orig.jpg'.format(int(time.time())))
    imutil.add_to_figure(img)
    print("Original Classification: {}".format(classifier_dataset.unformat_output(original_class)))

    print("Counterfactual Image:")
    img = decoder.predict(Z)[0]
    imutil.show(img, filename='{}_counterfactual_{}.jpg'.format(int(time.time()), selected_class))
    imutil.add_to_figure(img)
    print("Counterfactual Classification: {}".format(classifier_dataset.unformat_output(classification)))
    imutil.show_figure(filename='{}_counterfactual.jpg'.format(int(time.time())), resize_to=None)

    return trajectory
