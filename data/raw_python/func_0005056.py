def layers(self):
        '''Construct Keras input layers for the given transformer

        Returns
        -------
        layers : {field: keras.layers.Input}
            A dictionary of keras input layers, keyed by the corresponding
            field keys.
        '''
        from keras.layers import Input

        L = dict()
        for key in self.fields:
            L[key] = Input(name=key,
                           shape=self.fields[key].shape,
                           dtype=self.fields[key].dtype)

        return L