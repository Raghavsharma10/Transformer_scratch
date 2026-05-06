def handle_bail(self, bail):
        """Handle a bail line."""
        self._add_error(_("Bailed: {reason}").format(reason=bail.reason))