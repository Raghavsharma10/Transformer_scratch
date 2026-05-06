def available(self):
        ''' Check if a related database exists '''
        return self.db_name in map(
            lambda x: x['name'], self._db.get_database_list()
        )