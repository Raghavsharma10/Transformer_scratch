def delete(self):
        """Delete the draft."""
        response = self.session.request("delete:Message", [ self.message_id ])
        self.data = response
        return self