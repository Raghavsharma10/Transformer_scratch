def Pool(pool='AnyPool', **kwargs):
    '''
    Chooses between the different pools.
    If ``pool == 'AnyPool'``, chooses based on availability.

    '''

    if pool == 'MPIPool':
        return MPIPool(**kwargs)
    elif pool == 'MultiPool':
        return MultiPool(**kwargs)
    elif pool == 'SerialPool':
        return SerialPool(**kwargs)
    elif pool == 'AnyPool':
        if MPIPool.enabled():
            return MPIPool(**kwargs)
        elif MultiPool.enabled():
            return MultiPool(**kwargs)
        else:
            return SerialPool(**kwargs)
    else:
        raise ValueError('Invalid pool ``%s``.' % pool)