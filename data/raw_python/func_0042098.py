def create_crop(self, name, x, x2, y, y2):
        """
        Create a crop for this asset.
        """
        if self._can_crop():
            spec = get_image_cropper().create_crop(name, self.file, x=x,
                                                   x2=x2, y=y, y2=y2)
            ImageDetail.save_crop_spec(self, spec)