def getMultiple(self, layers, where="1 = 1", fields=[], srid='4326', layer_name_field=None):
        """
        Get a bunch of layers and concatenate them together into one. This is useful if you
        have a map with layers for, say, every year named stuff_2014, stuff_2013, stuff_2012. Etc.

        Optionally, you can stuff the source layer name into a field of your choosing.

        >>> arc.getMultiple([0, 3, 5], layer_name_field='layer_src_name')

        """
        features = []
        for layer in layers:
            get_fields = fields or self.enumerate_layer_fields(layer)
            this_layer = self.get(layer, where, get_fields, False, srid).get('features')
            if layer_name_field:
                descriptor = self.get_descriptor_for_layer(layer)
                layer_name = descriptor.get('name')
                for feature in this_layer:
                    feature['properties'][layer_name_field] = layer_name
            features += this_layer
        return {
            'type': "FeatureCollection",
            'features': features
        }