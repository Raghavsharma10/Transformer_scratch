def init(device_id=None, random_seed=None):
    """Initialize Hebel.

    This function creates a CUDA context, CUBLAS context and
    initializes and seeds the pseudo-random number generator.

    **Parameters:**
    
    device_id : integer, optional
        The ID of the GPU device to use. If this is omitted, PyCUDA's
        default context is used, which by default uses the fastest
        available device on the system. Alternatively, you can put the
        device id in the environment variable ``CUDA_DEVICE`` or into
        the file ``.cuda-device`` in the user's home directory.

    random_seed : integer, optional
        The seed to use for the pseudo-random number generator. If
        this is omitted, the seed is taken from the environment
        variable ``RANDOM_SEED`` and if that is not defined, a random
        integer is used as a seed.
    """

    if device_id is None:
        random_seed = _os.environ.get('CUDA_DEVICE')
    
    if random_seed is None:
        random_seed = _os.environ.get('RANDOM_SEED')

    global is_initialized
    if not is_initialized:
        is_initialized = True

        global context
        context.init_context(device_id)

        from pycuda import gpuarray, driver, curandom

        # Initialize memory pool
        global memory_pool
        memory_pool.init()

        # Initialize PRG
        global sampler
        sampler.set_seed(random_seed)

        # Initialize pycuda_ops
        from hebel import pycuda_ops
        pycuda_ops.init()