def get_error(self, error):
        """
        A helper function, gets standard information from the error.
        """
        error_type = type(error)
        if error.error_type == ET_CLIENT:
            error_type_name = 'Client'
        else:
            error_type_name = 'Server'
        return {
            'type': error_type_name,
            'name': error_type.__name__,
            'prefix': getattr(error_type, '__module__', ''),
            'message': unicode(error),
            'params': error.args,
        }