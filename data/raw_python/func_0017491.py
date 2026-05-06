def close(self, response):
        """Close connection to database."""
        LOGGER.info('Closing [%s]', os.getpid())
        if not self.database.is_closed():
            self.database.close()
        return response