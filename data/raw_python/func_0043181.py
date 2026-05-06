def send_preview(self): # pragma: no cover
        """Send a preview of this draft."""
        response = self.session.request("method:queuePreview", [ self.data ])
        self.data = response
        return self