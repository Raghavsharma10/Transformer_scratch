def get_value(self):
        """Returns the value of the constant."""
        if self.value is not_computed:
            self.value = self.value_provider()
            if self.value is not_computed:
                return None
        return self.value