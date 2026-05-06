def formatted_description(self):
        """ Returns a formatted description string (%s* tokens replaced) or None if unavailable """
        desc = self.description

        if desc:
            return desc.replace("%s1", self.formatted_value)
        else:
            return None