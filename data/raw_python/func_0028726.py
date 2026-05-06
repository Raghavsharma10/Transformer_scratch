def get_concrete_class(cls, class_name):
        """This method provides easier access to all writers inheriting Writer class

        :param class_name: name of the parser (name of the parser class which should be used)
        :type class_name: str
        :return: Writer subclass specified by parser_name
        :rtype: Writer subclass
        :raise ValueError:
        """
        def recurrent_class_lookup(cls):
            for cls in cls.__subclasses__():
                if lower(cls.__name__) == lower(class_name):
                    return cls
                elif len(cls.__subclasses__()) > 0:
                    r = recurrent_class_lookup(cls)
                    if r is not None:
                        return r
            return None

        cls = recurrent_class_lookup(cls)
        if cls:
            return cls
        else:
            raise ValueError("'class_name '%s' is invalid" % class_name)