def is_processed(self, db_versions):
        """Check if version is already applied in the database.

        :param db_versions:
        """
        return self.number in (v.number for v in db_versions if v.date_done)