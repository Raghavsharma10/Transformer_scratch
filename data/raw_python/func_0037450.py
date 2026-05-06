def create_all(self, checkfirst=True):
        """Creates all tables from models in the database
        
        :param bool checkfirst: Check if tables already exists
        """
        log.info('creating tables in %s', self.engine.url)
        models.Base.metadata.create_all(self.engine, checkfirst=checkfirst)