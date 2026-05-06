def create(self, db_name, **kwargs):
        """
        Construct a PostgresDatabase and create it on self
        """
        db = PostgresDatabase(
            db_name, host=self.host, port=self.port,
            superuser=self.superuser, **kwargs)
        db.ensure_user()
        db.create()
        return db