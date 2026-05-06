def worker(inqueue, outqueue, initializer=None, initargs=(), maxtasks=None):
    """ Custom worker for bundle operations

    :param inqueue:
    :param outqueue:
    :param initializer:
    :param initargs:
    :param maxtasks:
    :return:
    """
    from ambry.library import new_library
    from ambry.run import get_runconfig
    import traceback

    assert maxtasks is None or (type(maxtasks) == int and maxtasks > 0)

    put = outqueue.put
    get = inqueue.get

    if hasattr(inqueue, '_writer'):
        inqueue._writer.close()
        outqueue._reader.close()

    if initializer is not None:
        initializer(*initargs)

    try:
        task = get()
    except (EOFError, IOError):
        debug('worker got EOFError or IOError -- exiting')
        return

    if task is None:
        debug('worker got sentinel -- exiting')
        return

    job, i, func, args, kwds = task

    # func = mapstar = map(*args)

    # Since there is only one source build per process, we know the structure
    # of the args beforehand.
    mp_func = args[0][0]
    mp_args = list(args[0][1][0])

    library = new_library(get_runconfig())
    library.database.close()  # Maybe it is still open after the fork.
    library.init_debug()

    bundle_vid = mp_args[0]

    try:

        b = library.bundle(bundle_vid)
        library.logger = b.logger # So library logs to the same file as the bundle.

        b = b.cast_to_subclass()
        b.multi = True  # In parent it is a number, in child, just needs to be true to get the right logger template
        b.is_subprocess = True
        b.limited_run = bool(int(os.getenv('AMBRY_LIMITED_RUN', 0)))

        assert b._progress == None  # Don't want to share connections across processes

        mp_args[0] = b
        result = (True, [mp_func(*mp_args)])

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        b.error('Subprocess {} raised an exception: {}'.format(os.getpid(), e.message), False)
        b.error(tb, False)
        result = (False, e)

    assert result

    b.progress.close()
    library.close()

    try:
        put((job, i, result))
    except Exception as e:
        wrapped = MaybeEncodingError(e, result[1])
        debug("Possible encoding error while sending result: %s" % (wrapped))
        put((job, i, (False, wrapped)))