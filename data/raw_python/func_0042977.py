def __model_class(self, model_name):
        """ this method is used by the lru_cache, do not call directly """
        build_schema = deepcopy(self.definitions[model_name])
        return self.schema_class(build_schema, model_name)