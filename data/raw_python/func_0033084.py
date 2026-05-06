def copy(self, new_id=None, attribute_overrides={}):
        """
        Copies the DatabaseObject under the ID_KEY new_id.
        
        @param new_id: the value for ID_KEY of the copy; if this is none,
            creates the new object with a random ID_KEY
        @param attribute_overrides: dictionary of attribute names -> values that you would like to override with.
        """
        data = dict(self)
        data.update(attribute_overrides)
        if new_id is not None:
            data[ID_KEY] = new_id
            return self.create(data, path=self.PATH)
        else:
            del data[ID_KEY]
            return self.create(data, random_id=True, path=self.PATH)