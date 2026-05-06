def create_in_hdx(self):
        # type: () -> None
        """Check if resource view exists in HDX and if so, update it, otherwise create resource view

        Returns:
            None
        """
        self.check_required_fields()
        if not self._update_resource_view(log=True):
            self._save_to_hdx('create', 'title')