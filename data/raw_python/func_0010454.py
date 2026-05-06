def serialize_params(self, params, complex=False):
        """
        Serialize parameter names and values to a dict ready for urlencode()
        """
        if complex:
            # See climata.acis for an example implementation
            raise NotImplementedError("Cannot serialize %s!" % params)
        else:
            # Simpler queries can use traditional URL parameters
            return {
                self.get_url_param(key): ','.join(val)
                for key, val in params.items()
            }