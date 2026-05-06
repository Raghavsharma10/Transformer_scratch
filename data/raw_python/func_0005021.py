def layers(self):
        '''Construct Keras input layers for all feature transformers
        in the pump.

        Returns
        -------
        layers : {field: keras.layers.Input}
            A dictionary of keras input layers, keyed by the corresponding
            fields.
        '''

        layermap = dict()
        for operator in self.ops:
            if hasattr(operator, 'layers'):
                layermap.update(operator.layers())
        return layermap