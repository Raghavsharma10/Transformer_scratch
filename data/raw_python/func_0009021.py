def __gdal_dataset_default(self):
        """DiskReader implementation."""

        if not os.path.exists(self.file_name):
            return None
        if os.path.splitext(self.file_name)[1].lower() not in self.file_types:
            raise RuntimeError('Filename %s does not have extension type %s.' % (self.file_name, self.file_types))

        dataset = gdal.OpenShared(self.file_name, gdalconst.GA_ReadOnly)
        if dataset is None:
            raise ValueError('Dataset %s did not load properly.' % self.file_name)

        # Sanity checks.
        assert dataset.RasterCount > 0

        # Seems okay...
        return dataset