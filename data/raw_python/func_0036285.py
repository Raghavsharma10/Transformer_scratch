def keys(self):
        """Return a list of all keys in the dictionary.

        Returns:
            list of str: [key1,key2,...,keyN]
        """
        all_keys = [k.decode('utf-8') for k,v in self.rdb.hgetall(self.session_hash).items()]
        return all_keys