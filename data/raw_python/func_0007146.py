def validate(self, model_name, object):
        """Validate an object against its swagger model"""
        if model_name not in self.swagger_dict['definitions']:
            raise ValidationError("Swagger spec has no definition for model %s" % model_name)
        model_def = self.swagger_dict['definitions'][model_name]
        log.debug("Validating %s" % model_name)
        return validate_schema_object(self.spec, model_def, object)