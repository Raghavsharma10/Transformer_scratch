def update_layers(self):
        """
        Update layers for a service.
        """

        signals.post_save.disconnect(layer_post_save, sender=Layer)

        try:
            LOGGER.debug('Updating layers for service id %s' % self.id)
            if self.type == 'OGC:WMS':
                update_layers_wms(self)
            elif self.type == 'OGC:WMTS':
                update_layers_wmts(self)
            elif self.type == 'ESRI:ArcGIS:MapServer':
                update_layers_esri_mapserver(self)
            elif self.type == 'ESRI:ArcGIS:ImageServer':
                update_layers_esri_imageserver(self)
            elif self.type == 'Hypermap:WorldMapLegacy':
                update_layers_wm_legacy(self)
            elif self.type == 'Hypermap:WorldMap':
                update_layers_geonode_wm(self)
            elif self.type == 'Hypermap:WARPER':
                update_layers_warper(self)

        except:
            LOGGER.error('Error updating layers for service %s' % self.uuid)

        signals.post_save.connect(layer_post_save, sender=Layer)