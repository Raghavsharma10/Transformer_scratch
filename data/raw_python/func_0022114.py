def get_url_endpoint(self):
        """
        Returns the Hypermap endpoint for a layer.
        This endpoint will be the WMTS MapProxy endpoint, only for WM we use the original endpoint.
        """
        endpoint = self.url
        if self.type not in ('Hypermap:WorldMap',):
            endpoint = 'registry/%s/layer/%s/map/wmts/1.0.0/WMTSCapabilities.xml' % (
                self.catalog.slug,
                self.id
            )
        return endpoint