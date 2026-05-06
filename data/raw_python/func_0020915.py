def validate_spec(self, key, value):
        """Forbit updates of set identifier."""
        if self.spec and self.spec != value:
            raise OAISetSpecUpdateError("Updating spec is not allowed.")
        return value