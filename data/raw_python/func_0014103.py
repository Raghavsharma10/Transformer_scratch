def _set_attrs_to_values(self, response={}):
        """ 
        Set attributes to dictionary values so can access via dot notation.
        """
        for key in response.keys():
            setattr(self, key, response[key])