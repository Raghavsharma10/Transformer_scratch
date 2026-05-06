def create_alignment(self, x_align=0, y_align=0, x_scale=0, y_scale=0):
        """
        Function creates an alignment
        """
        align = Gtk.Alignment()
        align.set(x_align, y_align, x_scale, y_scale)
        return align