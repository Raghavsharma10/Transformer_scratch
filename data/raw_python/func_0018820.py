def find(self, text):
        """Print all substitutions that include the given text string."""
        for key, value in self:
            if (text in key) or (text in value):
                print(key, value)