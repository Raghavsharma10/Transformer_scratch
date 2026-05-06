def drop_all(self):
        """Drops all tables in the database"""
        log.info('dropping tables in %s', self.engine.url)
        self.session.commit()
        models.Base.metadata.drop_all(self.engine)
        self.session.commit()