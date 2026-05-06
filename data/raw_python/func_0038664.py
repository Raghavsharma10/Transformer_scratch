def expand_models(self, target, data):
        """
        Generates all objects from given data.
        """
        if isinstance(data, dict):
            data = data.values()
        for chunk in data:
            if target in chunk:
                yield self.init_target_object(target, chunk)
            else:
                for key, item in chunk.items():
                    yield self.init_single_object(key, item)