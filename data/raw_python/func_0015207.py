def create_gtk_grid(self, row_spacing=6, col_spacing=6, row_homogenous=False, col_homogenous=True):
        """
        Function creates a Gtk Grid with spacing
        and homogeous tags
        """
        grid_lang = Gtk.Grid()
        grid_lang.set_column_spacing(row_spacing)
        grid_lang.set_row_spacing(col_spacing)
        grid_lang.set_border_width(12)
        grid_lang.set_row_homogeneous(row_homogenous)
        grid_lang.set_column_homogeneous(col_homogenous)
        return grid_lang