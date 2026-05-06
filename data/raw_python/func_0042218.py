def create_crop(self, name, file_obj,
                    x=None, x2=None, y=None, y2=None):
        """
        Generate Version for an Image.
        value has to be a serverpath relative to MEDIA_ROOT.

        Returns the spec for the crop that was created.
        """

        if name not in self._registry:
            return

        file_obj.seek(0)
        im = Image.open(file_obj)
        config = self._registry[name]

        if x is not None and x2 and y is not None and y2 and not config.editable:
            # You can't ask for something special
            # for non editable images
            return

        im = config.rotate_by_exif(im)
        crop_spec = config.get_crop_spec(im, x=x, x2=x2, y=y, y2=y2)
        image = config.process_image(im, crop_spec=crop_spec)
        if image:
            crop_name = utils.get_size_filename(file_obj.name, name)
            self._save_file(image, crop_name)
            return crop_spec