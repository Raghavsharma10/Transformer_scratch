def enrich_app(self, name, value):
        '''
        Add a new property to the app (with setattr)

        Args:
            name (str): the name of the new property
            value (any): the value of the new property
        '''
        #Method shouldn't be added:  https://stackoverflow.com/a/28060251/3042398
        if type(value) == type(self.enrich_app):
            raise ValueError("enrich_app can't add method")

        setattr(self.app, name, value)