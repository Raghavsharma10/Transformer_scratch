def get_service_resources(cls, model):
        """ Get resource models by service model """
        key = cls.get_model_key(model)
        return cls.get_service_name_resources(key)