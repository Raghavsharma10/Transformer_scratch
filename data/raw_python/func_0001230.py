def __retrieve(self, key):
        ''' Retrieve file location from cache DB
        '''
        with self.get_conn() as conn:
            try:
                c = conn.cursor()
                if key is None:
                    c.execute("SELECT value FROM cache_entries WHERE key IS NULL")
                else:
                    c.execute("SELECT value FROM cache_entries WHERE key = ?", (key,))
                result = c.fetchone()
                if result is None or len(result) != 1:
                    getLogger().info("There's no entry with key={key}".format(key=key))
                    return None
                else:
                    return result[0]
            except:
                getLogger().exception("Cannot retrieve")
                return None