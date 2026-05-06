def get_descriptor_for_layer(self, layer):
        """
        Returns the standard JSON descriptor for the layer. There is a lot of
        usefule information in there.
        """
        if not layer in self._layer_descriptor_cache:
            params = {'f': 'pjson'}
            if self.token:
                params['token'] = self.token
            response = requests.get(self._build_request(layer), params=params)
            self._layer_descriptor_cache[layer] = response.json()
        return self._layer_descriptor_cache[layer]