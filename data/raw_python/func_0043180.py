def save(self):
        """Save current draft state."""
        response = self.session.request("save:Message", [ self.data ])
        self.data = response
        self.message_id = self.data["id"]
        return self