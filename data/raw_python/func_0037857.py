def get_adapter_for_persistent_model(self, persistent_model, rest_model=None):
        """
        :param persistent_model: instance of persistent model
        :param rest_model: specific REST model
        :return: the matching model adapter
        :rtype: ModelAdapter
        """
        persistent_signature = self.generate_signature(persistent_model)
        
        if persistent_signature in self._persistent_map:
            sub_map = self._persistent_map[persistent_signature]

            # return the first match if REST model was not specified
            if rest_model is None:
                return self._persistent_map[persistent_signature][self.DEFAULT_REST_ADAPTER]
            else:
                rest_sig = self.generate_signature(rest_model)
                if rest_sig in sub_map:
                    return self._persistent_map[persistent_signature][rest_sig]

        raise TypeError("No registered Data Adapter for class %s" % persistent_signature)