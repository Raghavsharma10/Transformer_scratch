def reproject_to_grid_coordinates(self, grid_coordinates, interp=gdalconst.GRA_NearestNeighbour):
        """ Reprojects data in this layer to match that in the GridCoordinates
        object. """
        source_dataset = self.grid_coordinates._as_gdal_dataset()
        dest_dataset = grid_coordinates._as_gdal_dataset()
        rb = source_dataset.GetRasterBand(1)
        rb.SetNoDataValue(NO_DATA_VALUE)
        rb.WriteArray(np.ma.filled(self.raster_data, NO_DATA_VALUE))

        gdal.ReprojectImage(source_dataset, dest_dataset,
                            source_dataset.GetProjection(),
                            dest_dataset.GetProjection(),
                            interp)
        dest_layer = self.clone_traits()
        dest_layer.grid_coordinates = grid_coordinates
        rb = dest_dataset.GetRasterBand(1)
        dest_layer.raster_data = np.ma.masked_values(rb.ReadAsArray(), NO_DATA_VALUE)
        return dest_layer