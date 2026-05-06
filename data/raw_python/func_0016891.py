def create_queue_wrapper(name, queue_size, fed_arrays, data_sources, *args, **kwargs):
    """
    Arguments
        name: string
            Name of the queue
        queue_size: integer
            Size of the queue
        fed_arrays: list
            array names that will be fed by this queue
        data_sources: dict
            (lambda/method, dtype) tuples, keyed on array names

    """

    qtype = SingleInputMultiQueueWrapper if 'count' in kwargs else QueueWrapper
    return qtype(name, queue_size, fed_arrays, data_sources, *args, **kwargs)