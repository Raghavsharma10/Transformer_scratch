def vcr(decorated_func=None, debug=False, overwrite=False, disabled=False,
        playback_only=False, tape_name=None):
    """
    Decorator for capturing and simulating network communication

    ``debug`` : bool, optional
        Enables debug mode.
    ``overwrite`` : bool, optional
        Will run vcr in recording mode - overwrites any existing vcrtapes.
    ``playback_only`` : bool, optional
        Will run vcr in playback mode - will not create missing vcrtapes.
    ``disabled`` : bool, optional
        Completely disables vcr - same effect as removing the decorator.
    ``tape_name`` : str, optional
        Use given custom file name instead of an auto-generated name for the
        tape file.
    """
    def _vcr_outer(func):
        """
        Wrapper around _vcr_inner allowing optional arguments on decorator
        """
        def _vcr_inner(*args, **kwargs):
            """
            The actual decorator doing a lot of monkey patching and auto magic
            """
            if disabled or VCRSystem.disabled:
                # execute decorated function without VCR
                return func(*args, **kwargs)

            # prepare VCR tape
            if func.__module__ == 'doctest':
                source_filename = func.__self__._dt_test.filename
                file_name = os.path.splitext(
                    os.path.basename(source_filename))[0]
                # check if a tests directory exists
                path = os.path.join(os.path.dirname(source_filename),
                                    'tests')
                if os.path.exists(path):
                    # ./test/vcrtapes/tape_name.vcr
                    path = os.path.join(os.path.dirname(source_filename),
                                        'tests', 'vcrtapes')
                else:
                    # ./vcrtapes/tape_name.vcr
                    path = os.path.join(os.path.dirname(source_filename),
                                        'vcrtapes')
                func_name = func.__self__._dt_test.name.split('.')[-1]
            else:
                source_filename = func.__code__.co_filename
                file_name = os.path.splitext(
                    os.path.basename(source_filename))[0]
                path = os.path.join(
                    os.path.dirname(source_filename), 'vcrtapes')
                func_name = func.__name__

            if tape_name:
                # tape file name is given - either full path is given or use
                # 'vcrtapes' directory
                if os.sep in tape_name:
                    temp = os.path.abspath(tape_name)
                    path = os.path.dirname(temp)
                    if not os.path.isdir(path):
                        os.makedirs(path)
                tape = os.path.join(path, '%s' % (tape_name))
            else:
                # make sure 'vcrtapes' directory exists
                if not os.path.isdir(path):
                    os.makedirs(path)
                # auto-generated file name
                tape = os.path.join(path, '%s.%s.vcr' % (file_name, func_name))

            # enable VCR
            with VCRSystem(debug=debug):
                # check for tape file and determine mode
                if not (playback_only or VCRSystem.playback_only) and (
                        not os.path.isfile(tape) or
                        overwrite or VCRSystem.overwrite):
                    # record mode
                    if PY2:
                        msg = 'VCR records only in PY3 to be backward ' + \
                              'compatible with PY2 - skipping VCR ' + \
                              'mechanics for %s'
                        warnings.warn(msg % (func.__name__))
                        # disable VCR
                        VCRSystem.stop()
                        # execute decorated function without VCR
                        return func(*args, **kwargs)
                    if VCRSystem.debug:
                        print('\nVCR RECORDING (%s) ...' % (func_name))
                    VCRSystem.status = VCR_RECORD
                    # execute decorated function
                    value = func(*args, **kwargs)
                    # check if vcr is actually used at all
                    if len(VCRSystem.playlist) == 0:
                        msg = 'no socket activity - @vcr unneeded for %s'
                        msg = msg % (func.__name__)
                        if VCRSystem.raise_if_not_needed:
                            raise Exception(msg)
                        else:
                            warnings.warn(msg)
                    else:
                        # remove existing tape
                        try:
                            os.remove(tape)
                        except OSError:
                            pass
                        # write playlist to file
                        with gzip.open(tape, 'wb') as fh:
                            pickle.dump(VCRSystem.playlist, fh, protocol=2)
                else:
                    # playback mode
                    if VCRSystem.debug:
                        print('\nVCR PLAYBACK (%s) ...' % (func_name))
                    VCRSystem.status = VCR_PLAYBACK
                    # if playback is requested and tape is missing: raise!
                    if not os.path.exists(tape):
                        msg = 'Missing VCR tape file for playback: {}'
                        raise IOError(msg.format(tape))
                    # load playlist
                    try:
                        with gzip.open(tape, 'rb') as fh:
                            VCRSystem.playlist = pickle.load(fh)
                    except OSError:
                        # support for older uncompressed tapes
                        with open(tape, 'rb') as fh:
                            VCRSystem.playlist = pickle.load(fh)
                    if VCRSystem.debug:
                        print('Loaded playlist:')
                        for i, item in enumerate(VCRSystem.playlist):
                            print('{:3d}: {} {} {}'.format(i, *item))
                        print()
                    # execute decorated function
                    value = func(*args, **kwargs)

            return value

        return _vcr_inner

    if decorated_func is None:
        # without arguments
        return _vcr_outer
    else:
        # with arguments
        return _vcr_outer(decorated_func)