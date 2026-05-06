def project_name_changed(self, widget, data=None):
        """
        Function controls whether run button is enabled
        """
        if widget.get_text() != "":
            self.run_btn.set_sensitive(True)
        else:
            self.run_btn.set_sensitive(False)
        self.update_full_label()