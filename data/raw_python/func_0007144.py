def model_to_json(self, object, cleanup=True):
        """Take a model instance and return it as a json struct"""
        model_name = type(object).__name__
        if model_name not in self.swagger_dict['definitions']:
            raise ValidationError("Swagger spec has no definition for model %s" % model_name)
        model_def = self.swagger_dict['definitions'][model_name]
        log.debug("Marshalling %s into json" % model_name)
        m = marshal_model(self.spec, model_def, object)
        if cleanup:
            self.cleanup_model(m)
        return m