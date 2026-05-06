def init_db(self):
        """
            This function configures the database used for models to make
            the configuration parameters.
        """
        # get the database url from the configuration
        db_url = self.config.get('database_url', 'sqlite:///nautilus.db')
        # configure the nautilus database to the url
        nautilus.database.init_db(db_url)