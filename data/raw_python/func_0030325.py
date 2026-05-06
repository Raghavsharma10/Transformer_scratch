def filesessionmaker(sessionmaker, file_manager, file_managers=None):
    u'''Wrapper of session maker adding link to a FileManager instance
    to session.::

        file_manager = FileManager(cfg.TRANSIENT_ROOT,
                                   cfg.PERSISTENT_ROOT)
        filesessionmaker(sessionmaker(...), file_manager)
    '''

    registry = WeakKeyDictionary()

    if file_managers:
        for k, v in six.iteritems(file_managers):
            if isinstance(k, FileAttribute):
                raise NotImplementedError()
            registry[k] = v

    def find_file_manager(self, target):
        if isinstance(target, FileAttribute):
            assert hasattr(target, 'class_')
            target = target.class_
        else:
            if not inspect.isclass(target):
                target = type(target)

        assert hasattr(target, 'metadata')
        assert class_mapper(target) is not None
        if target in registry:
            return registry[target]
        if target.metadata in registry:
            return registry[target.metadata]
        return file_manager

    def session_maker(*args, **kwargs):
        session = sessionmaker(*args, **kwargs)
        # XXX in case we want to use session manager somehow bound 
        #     to request environment. For example, to generate user-specific
        #     URLs.
        #session.file_manager = \
        #        kwargs.get('file_manager', file_manager)
        session.file_manager = file_manager
        session.find_file_manager = six.create_bound_method(
                                            find_file_manager,
                                            session)

        return session
    return session_maker