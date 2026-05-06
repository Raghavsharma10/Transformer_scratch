def __insert(self, key, value):
        '''
        Insert a new key to database
        '''
        if key in self:
            getLogger().warning("Cache entry exists, cannot insert a new entry with key='{key}'".format(key=key))
            return False
        with self.get_conn() as conn:
            try:
                c = conn.cursor()
                c.execute("INSERT INTO cache_entries (key, value) VALUES (?,?)", (key, value))
                conn.commit()
                return True
            except Exception as e:
                # NOTE: A cache error can be forgiven, no?
                getLogger().debug("Cache Error: Cannot insert | Detail = %s" % (e,))
                return False