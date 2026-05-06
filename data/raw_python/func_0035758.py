def put_overlay(self, overlay_name, overlay):
        """Store the overlay."""
        logger.debug("Putting overlay: {}".format(overlay_name))
        key = self.get_overlay_key(overlay_name)
        text = json.dumps(overlay, indent=2)
        self.put_text(key, text)