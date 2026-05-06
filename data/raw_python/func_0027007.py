def get_name_for_model(cls, model):
        """ Get a name for given class or model:
            -- it's a service type for a service
            -- it's a <service_type>.<resource_model_name> for a resource
        """
        key = cls.get_model_key(model)
        model_str = cls._get_model_str(model)
        service = cls._registry[key]
        if model_str in service['resources']:
            return '{}.{}'.format(service['name'], service['resources'][model_str]['name'])
        else:
            return service['name']