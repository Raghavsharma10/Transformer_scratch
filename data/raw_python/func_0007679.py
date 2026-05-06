def _softmax(x, dim):
    """Computes softmax along a specified dim. Keras currently lacks this feature.
    """

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.nn.softmax(x, dim)
    elif K.backend() is 'cntk':
        import cntk
        return cntk.softmax(x, dim)
    elif K.backend() == 'theano':
        # Theano cannot softmax along an arbitrary dim.
        # So, we will shuffle `dim` to -1 and un-shuffle after softmax.
        perm = np.arange(K.ndim(x))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x_perm = K.permute_dimensions(x, perm)
        output = K.softmax(x_perm)

        # Permute back
        perm[dim], perm[-1] = perm[-1], perm[dim]
        output = K.permute_dimensions(x, output)
        return output
    else:
        raise ValueError("Backend '{}' not supported".format(K.backend()))