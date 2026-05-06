def _scene_centroid(self):
        """ Compute image center coordinates
        :return: Tuple of image center in lat, lon
        """
        ul_lat = self.corner_ul_lat_product
        ll_lat = self.corner_ll_lat_product
        ul_lon = self.corner_ul_lon_product
        ur_lon = self.corner_ur_lon_product
        lat = (ul_lat + ll_lat) / 2.
        lon = (ul_lon + ur_lon) / 2.

        return lat, lon