def delete(self):
        """
        Deletes the current instance. This assumes that we know what we're
        doing, and have a primary key in our data already. If this is a new
        instance, then we'll let the user know with an Exception
        """
        if self._new:
            raise Exception("This is a new object, %s not in data, \
indicating this entry isn't stored." % self.primaryKey)

        r.table(self.table).get(self._data[self.primaryKey]) \
            .delete(durability=self.durability).run(self._conn)
        return True