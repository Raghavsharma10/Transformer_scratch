def get_instances(self, object_specs, version=None):
        """Get the cached native representation for one or more objects.

        Keyword arguments:
        object_specs - A sequence of triples (model name, pk, obj):
        - model name - the name of the model
        - pk - the primary key of the instance
        - obj - the instance, or None to load it
        version - The cache version to use, or None for default

        To get the 'new object' representation, set pk and obj to None

        Return is a dictionary:
        key - (model name, pk)
        value - (native representation, pk, object or None)
        """
        ret = dict()
        spec_keys = set()
        cache_keys = []
        version = version or self.default_version

        # Construct all the cache keys to fetch
        for model_name, obj_pk, obj in object_specs:
            assert model_name
            assert obj_pk

            # Get cache keys to fetch
            obj_key = self.key_for(version, model_name, obj_pk)
            spec_keys.add((model_name, obj_pk, obj, obj_key))
            cache_keys.append(obj_key)

        # Fetch the cache keys
        if cache_keys and self.cache:
            cache_vals = self.cache.get_many(cache_keys)
        else:
            cache_vals = {}

        # Use cached representations, or recreate
        cache_to_set = {}
        for model_name, obj_pk, obj, obj_key in spec_keys:

            # Load cached objects
            obj_val = cache_vals.get(obj_key)
            obj_native = json.loads(obj_val) if obj_val else None

            # Invalid or not set - load from database
            if not obj_native:
                if not obj:
                    loader = self.model_function(model_name, version, 'loader')
                    obj = loader(obj_pk)
                serializer = self.model_function(
                    model_name, version, 'serializer')
                obj_native = serializer(obj) or {}
                if obj_native:
                    cache_to_set[obj_key] = json.dumps(obj_native)

            # Get fields to convert
            keys = [key for key in obj_native.keys() if ':' in key]
            for key in keys:
                json_value = obj_native.pop(key)
                name, value = self.field_from_json(key, json_value)
                assert name not in obj_native
                obj_native[name] = value

            if obj_native:
                ret[(model_name, obj_pk)] = (obj_native, obj_key, obj)

        # Save any new cached representations
        if cache_to_set and self.cache:
            self.cache.set_many(cache_to_set)

        return ret