def _set_properties(self, data):
        """
        set the properties of the app model by the given data dict
        """
        for property in data.keys():
            if property in vars(self):
                setattr(self, property, data[property])