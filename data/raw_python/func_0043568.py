def modified_data_decorator(function):
    """
    Decorator to initialise the modified_data if necessary. To be used in list functions
    to modify the list
    """

    @wraps(function)
    def func(self, *args, **kwargs):
        """Decorator function"""
        if not self.get_read_only() or not self.is_locked():
            self.initialise_modified_data()
            return function(self, *args, **kwargs)
        return lambda: None

    return func