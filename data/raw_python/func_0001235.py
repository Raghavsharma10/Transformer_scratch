def __retrieve_internal_blob(self, key):
        ''' Retrieve file location from cache DB
        '''
        logger = getLogger()
        with self.get_conn() as conn:
            try:
                c = conn.cursor()
                if key is None:
                    c.execute("SELECT compressed, blob_data FROM blob_entries WHERE KEY IS NULL")
                else:
                    c.execute("SELECT compressed, blob_data FROM blob_entries WHERE KEY = ?", (key,))
                result = c.fetchone()
                if not result:
                    logger.debug("There's no blob entry with key={key}".format(key=key))
                    logger.debug("result = {res}".format(res=result))
                    return None
                else:
                    compressed, blob_data = result
                    logger.debug("retrieving internal BLOB (key={key} | len={ln} | compressed={c})".format(key=key, ln=len(blob_data), c=compressed))
                    return blob_data if not compressed else zlib.decompress(blob_data)
            except:
                getLogger().exception("Cannot retrieve internal blob (key={})".format(key))
                return None
            return True