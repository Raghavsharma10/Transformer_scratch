def get_formats(self):
        """
        Returns a dictionary of format options keyed by data kind.

        .. code-block:: python

            {
                "vector": {
                    "application/x-ogc-gpkg": "GeoPackage",
                    "application/x-zipped-shp": "Shapefile",
                    #...
                },
                "table": {
                    "text/csv": "CSV (text/csv)",
                    "application/x-ogc-gpkg": "GeoPackage",
                    #...
                },
                "raster": {
                    "image/jpeg": "JPEG",
                    "image/jp2": "JPEG2000",
                    #...
                },
                "grid": {
                    "application/x-ogc-aaigrid": "ASCII Grid",
                    "image/tiff;subtype=geotiff": "GeoTIFF",
                    #...
                },
                "rat": {
                    "application/x-erdas-hfa": "ERDAS Imagine",
                    #...
                }
            }

        :rtype: dict
        """
        format_opts = self._options()['actions']['POST']['formats']['children']
        r = {}
        for kind, kind_opts in format_opts.items():
            r[kind] = {c['value']: c['display_name'] for c in kind_opts['choices']}
        return r