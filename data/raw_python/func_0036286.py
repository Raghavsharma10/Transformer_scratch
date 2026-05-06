def values(self):
        """Returns a list of all values in the dictionary.

        Returns:
            list of str: [value1,value2,...,valueN]
        """
        all_values = [v.decode('utf-8') for k,v in self.rdb.hgetall(self.session_hash).items()]
        return all_values