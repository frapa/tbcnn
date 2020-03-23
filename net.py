import tensorflow as tf

def generate_convolutional_block(inp, filters, length=2, pool=True, stride=1):
    "Generates a convolutional block, with a couple of simple options"

    output = inp

    for i in range(length):
        # convolution
        output = tf.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
        )(output)

        # batch normalization
        output = tf.layers.batch_normalization(output)

        # ReLU
        output = tf.nn.relu(output)

    parallel = tf.layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=stride**length,
        padding='same',
        kernel_initializer=tf.keras.initializers.he_normal(),
    )(inp)

    # batch normalization
    parallel = tf.layers.batch_normalization(parallel)

    # ReLU
    parallel = tf.nn.relu(parallel)

    output = (output + parallel) / 2

    if pool:
        output = tf.layers.MaxPooling2D(
            pool_size=3,
            strides=2,
        )(output)
    
    return output

def generate_network(size=512, width=1):
    "Generates a tensorflow graph for the network and returns is"

    inp = tf.placeholder(tf.float32, [None, size, size, 1], name='input')
    labels = tf.placeholder(tf.int32, [None], name='labels')

    # First convolutiona block with only one 
    output = generate_convolutional_block(inp, filters=16*width, stride=2)

    # 3 "normal" convolutional blocks
    output = generate_convolutional_block(output, filters=32*width)
    output = generate_convolutional_block(output, filters=48*width)
    output = generate_convolutional_block(output, filters=64*width)

    # last convolutional block without pooling
    output = generate_convolutional_block(output, filters=80*width, pool=False)

    # Global average pooling
    output = tf.reduce_mean(output, axis=[1,2], name='gap')
    # output = tf.layers.flatten(output, name='flatten')

    # Dense layer for the output, with softmax activation
    logits = tf.layers.Dense(
        units=2, # 2 outputs
        kernel_initializer=tf.keras.initializers.he_normal(),
        name='logits',
    )(output)

    probabilities = tf.nn.softmax(logits, name='probabilities')
    classes = tf.argmax(logits, axis=1, name='classes')

    return inp, labels, {
        'logits': logits,
        'probabilities': probabilities,
        'classes': classes,
    }

def generate_functions(inp, labels, output):
    """Generates functions like error, accuracy and train,
    that are used for training and testing the network"""

    error = tf.losses.sparse_softmax_cross_entropy(labels, output['logits'])

    optimizer = tf.train.AdamOptimizer(learning_rate=8e-5)
    train = optimizer.minimize(error)

    metrics = {
        'accuracy': tf.metrics.accuracy(labels, output['classes']),
        'AUC': tf.metrics.auc(labels, output['probabilities'][:,1]),
    }

    return error, train, metrics
