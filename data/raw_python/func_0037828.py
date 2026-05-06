def current(cls):
        """Get the current event loop singleton object.
        """
        try:
            return _tls.loop
        except AttributeError:
            # create loop only for main thread
            if threading.current_thread().name == 'MainThread':
                _tls.loop = cls()
                return _tls.loop
            raise RuntimeError('there is no event loop created in the current thread')