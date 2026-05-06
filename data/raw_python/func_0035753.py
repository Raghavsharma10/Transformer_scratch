def get_overlay(self, overlay_name):
        """Return overlay as a dictionary."""
        logger.debug("Getting overlay: {}".format(overlay_name))
        overlay_key = self.get_overlay_key(overlay_name)
        text = self.get_text(overlay_key)
        return json.loads(text)