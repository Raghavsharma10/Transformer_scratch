def change_db(self, db, user=None):
        """Change connect database."""
        # Get original config and change database key
        config = self._config
        config['database'] = db
        if user:
            config['user'] = user
        self.database = db

        # Close current database connection
        self._disconnect()

        # Reconnect to the new database
        self._connect(config)