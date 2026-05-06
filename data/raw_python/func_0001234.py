def __delete_internal_blob(self, key):
        ''' This method will insert blob data to blob table
        '''
        with self.get_conn() as conn:
            conn.isolation_level = None
            try:
                c = conn.cursor()
                c.execute("BEGIN")
                if key is None:
                    c.execute("DELETE FROM cache_entries WHERE key IS NULL")
                    c.execute("DELETE FROM blob_entries WHERE KEY IS NULL")
                else:
                    c.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    c.execute("DELETE FROM blob_entries WHERE KEY = ?", (key,))
                c.execute("COMMIT")
            except:
                getLogger().debug("Cannot delete")
                return False
            return True