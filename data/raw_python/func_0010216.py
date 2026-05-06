def FloatProperty(name, default=0.0, readonly=False, docs=None):
    '''
    :name: string - property name
    :default: float - property default value
    :readonly: boolean - if True, setter method is NOT generated

    Returns a property object that can be used to initialize a
    class instance variable as a property.
    '''

    private_name = '_' + name

    def getf(self):
        if not hasattr(self, private_name):
            setattr(self, private_name, default)
        return getattr(self, private_name)

    if readonly:
        setf = None
    else:
        def setf(self, newValue):
            def epsilon_set(v):
                # epsilon_set: creates a float from v unless that
                #              float is less than epsilon, which will
                #              be considered effectively zero.
                fv = float(v)
                return 0.0 if nearly_zero(fv) else fv

            try:
                setattr(self, private_name, epsilon_set(newValue))
                return
            except TypeError:
                pass

            if isinstance(newValue, collections.Mapping):
                try:
                    setattr(self, private_name, epsilon_set(newValue[name]))
                except KeyError:
                    pass
                return

            if isinstance(newValue, collections.Iterable):
                try:
                    setattr(self, private_name, epsilon_set(newValue[0]))
                    return
                except (IndexError, TypeError):
                    pass

            try:
                mapping = vars(newValue)
                setattr(self, private_name, epsilon_set(mapping[name]))
                return
            except (TypeError, KeyError):
                pass

            if newValue is None:
                setattr(self, private_name, epsilon_set(default))
                return

            raise ValueError(newValue)

    return property(getf, setf, None, docs)