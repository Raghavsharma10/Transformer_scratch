def generateThumbnail(self):
        """Generates a square thumbnail"""
        image = pilImage.open(ROOT / self.source.name)
        box, width, height = cropBox(self.width, self.height)

        # Resize
        image.thumbnail((width, height), pilImage.ANTIALIAS)
        # Crop from center
        box = cropBox(*image.size)[0]
        image = image.crop(box)
        # save
        self.thumbnail = self.source.name.replace(self.hash, '__{}'.format(self.hash))
        image.save(ROOT / self.thumbnail.name)