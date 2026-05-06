def clear_ortho_view_set(self):
        """stub"""
        if (self.get_ortho_view_set_metadata().is_read_only() or
                self.get_ortho_view_set_metadata().is_required()):
            raise NoAccess()
        self.clear_file('frontView')
        self.clear_file('sideView')
        self.clear_file('topView')