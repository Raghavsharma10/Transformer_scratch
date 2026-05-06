def get_ceiling_cloud_layer(self):
        """
        Returns the lowest layer of broken or overcast clouds.
        :rtype: CloudLayer|None
        """
        lowest_layer = None
        for layer in self.cloud_layers:
            if layer.coverage not in [CloudLayer.BROKEN, CloudLayer.OVERCAST]:
                continue
            if lowest_layer is None:
                lowest_layer = layer
                continue
            if layer.height > lowest_layer.height:
                continue
            if layer.height < lowest_layer.height or \
                    lowest_layer.get_coverage_percentage() < layer.get_coverage_percentage():
                lowest_layer = layer
        return lowest_layer