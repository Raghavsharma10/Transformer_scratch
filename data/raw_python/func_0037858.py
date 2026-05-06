def get_adapter_for_rest_model(self, rest_model):
        """
        :param rest_model: instance of REST model
        :return: the matching model adapter
        :rtype: ModelAdapter
        """
        class_signature = self.generate_signature(rest_model)
        
        if class_signature not in self._rest_map:
            raise TypeError("No registered Data Adapter for class %s" % class_signature)

        return self._rest_map[class_signature]