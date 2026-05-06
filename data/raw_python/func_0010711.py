def patch_celery():
    """ Monkey patch Celery to use cloudpickle instead of pickle. """
    registry = serialization.registry
    serialization.pickle = cloudpickle
    registry.unregister('pickle')
    registry.register('pickle', cloudpickle_dumps, cloudpickle_loads,
                      content_type='application/x-python-serialize',
                      content_encoding='binary')

    import celery.worker as worker
    import celery.concurrency.asynpool as asynpool
    worker.state.pickle = cloudpickle
    asynpool._pickle = cloudpickle

    import billiard.common
    billiard.common.pickle = cloudpickle
    billiard.common.pickle_dumps = cloudpickle_dumps
    billiard.common.pickle_loads = cloudpickle_loads