def __delete(self, key):
        ''' Delete file key from database
        '''
        with self.get_conn() as conn:
            try:
                c = conn.cursor()
                c.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
            except:
                getLogger().exception("Cannot delete")
                return None