def rest_name(cls):
        """ Represents a singular REST name
        """
        if cls.__name__ == "NURESTRootObject" or cls.__name__ == "NURESTObject":
            return "Not Implemented"

        if cls.__rest_name__ is None:
            raise NotImplementedError('%s has no defined name. Implement rest_name property first.' % cls)

        return cls.__rest_name__