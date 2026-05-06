def _deps_only_toggled(self, widget, data=None):
        """
        Function deactivate options in case of deps_only and opposite
        """
        active = widget.get_active()
        self.dir_name.set_sensitive(not active)
        self.entry_project_name.set_sensitive(not active)
        self.dir_name_browse_btn.set_sensitive(not active)
        self.run_btn.set_sensitive(active or not self.project_name_shown or self.entry_project_name.get_text() != "")