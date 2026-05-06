def resource_name(cls):
        """ Represents the resource name
        """
        if cls.__name__ == "NURESTRootObject" or cls.__name__ == "NURESTObject":
            return "Not Implemented"

        if cls.__resource_name__ is None:
            raise NotImplementedError('%s has no defined resource name. Implement resource_name property first.' % cls)

        return cls.__resource_name__