def list_view_changed(self, widget, event, data=None):
        """
        Function shows last rows.
        """
        adj = self.scrolled_window.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())