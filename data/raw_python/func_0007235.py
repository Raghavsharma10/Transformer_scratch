def json_to_model(self, model_name, j, validate=False):
        """Take a json strust and a model name, and return a model instance"""
        if validate:
            self.api_spec.validate(model_name, j)
        return self.api_spec.json_to_model(model_name, j)