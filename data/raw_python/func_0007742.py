def validate(options):
        """ Validates the application of this backend to a given metadata 
        """
        try:
            if options.backends.index('modelinstance') > options.backends.index('model'):
                raise Exception("Metadata backend 'modelinstance' must come before 'model' backend")
        except ValueError:
            raise Exception("Metadata backend 'modelinstance' must be installed in order to use 'model' backend")