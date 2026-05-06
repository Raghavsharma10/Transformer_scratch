def create_box(self, orientation=Gtk.Orientation.HORIZONTAL, spacing=0):
        """
            Function creates box. Based on orientation
            it can be either HORIZONTAL or VERTICAL
        """
        h_box = Gtk.Box(orientation=orientation, spacing=spacing)
        h_box.set_homogeneous(False)
        return h_box