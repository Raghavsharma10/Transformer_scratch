def load_mnist():
    '''Load the MNIST digits dataset.'''
    mnist = skdata.mnist.dataset.MNIST()
    mnist.meta  # trigger download if needed.

    def arr(n, dtype):
        arr = mnist.arrays[n]
        return arr.reshape((len(arr), -1)).astype(dtype)
    train_images = arr('train_images', np.float32) / 128 - 1
    train_labels = arr('train_labels', np.uint8)
    return ((train_images[:50000], train_labels[:50000, 0]),
            (train_images[50000:], train_labels[50000:, 0]))