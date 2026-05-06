def change(self, key, value):
        """Update any other attribute on the build object"""
        self.obj[key] = value
        self.changes.append("Updating build:{}.{}={}"
                            .format(self.obj['name'], key, value))
        return self