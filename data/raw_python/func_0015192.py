def create_image(self, image_name=None, scale_ratio=1, window=None):
        """
            The function creates a image from name defined in image_name
        """
        size = 48 * scale_ratio
        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(image_name, -1, size, True)
        image = Gtk.Image()

        # Creating the cairo surface is necessary for proper scaling on HiDPI
        try:
            surface = Gdk.cairo_surface_create_from_pixbuf(pixbuf, scale_ratio, window)
            image.set_from_surface(surface)

        # Fallback for GTK+ older than 3.10
        except AttributeError:
            image.set_from_pixbuf(pixbuf)

        return image