def load_name(absolute_name: str):
        """Load an object based on an absolute, dotted name"""
        path = absolute_name.split('.')
        try:
            __import__(absolute_name)
        except ImportError:
            try:
                obj = sys.modules[path[0]]
            except KeyError:
                raise ModuleNotFoundError('No module named %r' % path[0])
            else:
                for component in path[1:]:
                    try:
                        obj = getattr(obj, component)
                    except AttributeError as err:
                        raise ConfigurationError(what='no such object %r: %s' % (absolute_name, err))
                return obj
        else:  # ImportError is not raised if ``absolute_name`` points to a valid module
            return sys.modules[absolute_name]