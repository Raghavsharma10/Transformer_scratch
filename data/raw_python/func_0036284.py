def items(self):
        """Return a list of all the key, value pair tuples in the dictionary.

        Returns:
            list of tuples: [(key1,value1),(key2,value2),...,(keyN,valueN)]
        """
        all_items = [(k.decode('utf-8'),v.decode('utf-8')) for k,v in self.rdb.hgetall(self.session_hash).items()]
        return all_items