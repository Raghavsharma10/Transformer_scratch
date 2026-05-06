def status(self, key, value):
        """Update the status of a build"""
        value = value.lower()
        if value not in valid_statuses:
            raise ValueError("Build Status must have a value from:\n{}".format(", ".join(valid_statuses)))

        self.obj['status'][key] = value
        self.changes.append("Updating build:{}.status.{}={}"
                            .format(self.obj['name'], key, value))
        return self