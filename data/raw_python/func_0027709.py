def persistentValues(self):
        """
        Return a dictionary of all attributes which will be/have been/are being
        stored in the database.
        """
        return dict((k, getattr(self, k)) for (k, attr) in self.getSchema())