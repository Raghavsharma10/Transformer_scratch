def _load_module(module):
    """Ensures that a module, and its parents, are properly loaded

    """
    modclass = type(module)
    # We only take care of our own LazyModule instances
    if not issubclass(modclass, LazyModule):
        raise TypeError("Passed module is not a LazyModule instance.")
    with _ImportLockContext():
        parent, _, modname = module.__name__.rpartition('.')
        logger.debug("loading module {}".format(modname))
        # We first identify whether this is a loadable LazyModule, then we
        # strip as much of lazy_import behavior as possible (keeping it cached,
        # in case loading fails and we need to reset the lazy state).
        if not hasattr(modclass, '_lazy_import_error_msgs'):
            # Alreay loaded (no _lazy_import_error_msgs attr). Not reloading.
            return
        # First, ensure the parent is loaded (using recursion; *very* unlikely
        # we'll ever hit a stack limit in this case).
        modclass._LOADING = True
        try:
            if parent:
                logger.debug("first loading parent module {}".format(parent))
                setattr(sys.modules[parent], modname, module)
            if not hasattr(modclass, '_LOADING'):
                logger.debug("Module {} already loaded by the parent"
                             .format(modname))
                # We've been loaded by the parent. Let's bail.
                return
            cached_data = _clean_lazymodule(module)
            try:
                # Get Python to do the real import!
                reload_module(module)           
            except:
                # Loading failed. We reset our lazy state.
                logger.debug("Failed to load module {}. Resetting..."
                             .format(modname))
                _reset_lazymodule(module, cached_data)
                raise
            else:
                # Successful load
                logger.debug("Successfully loaded module {}".format(modname))
                delattr(modclass, '_LOADING')
                _reset_lazy_submod_refs(module)

        except (AttributeError, ImportError) as err:
            logger.debug("Failed to load {}.\n{}: {}"
                         .format(modname, err.__class__.__name__, err))
            logger.lazy_trace()
            # Under Python 3 reloading our dummy LazyModule instances causes an
            # AttributeError if the module can't be found. Would be preferrable
            # if we could always rely on an ImportError. As it is we vet the
            # AttributeError as thoroughly as possible.
            if ((six.PY3 and isinstance(err, AttributeError)) and not
                err.args[0] == "'NoneType' object has no attribute 'name'"):
                # Not the AttributeError we were looking for.
                raise
            msg = modclass._lazy_import_error_msgs['msg']
            raise_from(ImportError(
                msg.format(**modclass._lazy_import_error_strings)), None)