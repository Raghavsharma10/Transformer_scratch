def add_attrs(self, *args, _order=[], **kwargs):
        """ Adds attributes to the __repr__ string
            @order: optional #list containing order to display kwargs
        """
        for arg in args:
            if isinstance(arg, (tuple, list)):
                key, color = arg
                self.attrs[key] = (None, color)
            else:
                self.attrs[arg] = (None, None)
        if not _order:
            for key, value in kwargs.items():
                self.attrs[key] = (value, None)
        else:
            for key in _order:
                self.attrs[key] = (kwargs[key], None)