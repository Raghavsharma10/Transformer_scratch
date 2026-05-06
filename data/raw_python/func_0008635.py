def paste(self, other):
        """Return a new Image with the given image pasted on top.

        This image will show through transparent areas of the given image.

        """
        r, g, b, alpha = other.pil_image.split()
        pil_image = self.pil_image.copy()
        pil_image.paste(other.pil_image, mask=alpha)
        return kurt.Image(pil_image)