def __insert_internal_blob(self, key, blob, compressed=True):
        ''' This method will insert blob data to blob table
        '''
        with self.get_conn() as conn:
            conn.isolation_level = None
            c = conn.cursor()
            try:
                compressed_flag = 1 if compressed else 0
                if compressed:
                    blob = zlib.compress(blob)
                c.execute("BEGIN")
                c.execute("INSERT INTO cache_entries (key, value) VALUES (?,?)", (key, JiCache.INTERNAL_BLOB))
                c.execute("INSERT INTO blob_entries (key, compressed, blob_data) VALUES (?,?,?)", (key, compressed_flag, sqlite3.Binary(blob),))
                c.execute("COMMIT")
                return True
            except:
                getLogger().debug("Cannot insert")
                return False