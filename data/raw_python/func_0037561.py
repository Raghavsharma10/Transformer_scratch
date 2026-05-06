def _updateable(self, obj):
        """Return True if there are available updates."""
        if obj.latest_version is None or obj.is_editable:
            return None
        else:
            return obj.latest_version != obj.current_version