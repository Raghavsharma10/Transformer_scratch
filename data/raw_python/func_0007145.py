def json_to_model(self, model_name, j):
        """Take a json strust and a model name, and return a model instance"""
        if model_name not in self.swagger_dict['definitions']:
            raise ValidationError("Swagger spec has no definition for model %s" % model_name)
        model_def = self.swagger_dict['definitions'][model_name]
        log.debug("Unmarshalling json into %s" % model_name)
        return unmarshal_model(self.spec, model_def, j)