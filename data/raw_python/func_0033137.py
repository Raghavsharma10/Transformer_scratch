def authenticate(self, username, password, db=None):
        """ Authenticates the MongoClient with the passed username and password """
        if db is None:
            return self.get_connection().admin.authenticate(username, password)
        return self.get_connection()[db].authenticate(username, password)