def load_tf_lib():
    """ Load the tensorflow library """
    from os.path import join as pjoin
    import pkg_resources

    import tensorflow as tf

    path = pjoin('ext', 'rime.so')
    rime_lib_path = pkg_resources.resource_filename("montblanc", path)
    return tf.load_op_library(rime_lib_path)