def save(self):
        """
        If an id exists in the database, we assume we'll update it, and if not
        then we'll insert it. This could be a problem with creating your own
        id's on new objects, however luckily, we keep track of if this is a new
        object through a private _new variable, and use that to determine if we
        insert or update.
        """
        if not self._new:
            data = self._data.copy()
            ID = data.pop(self.primaryKey)
            reply = r.table(self.table).get(ID) \
                .update(data,
                        durability=self.durability,
                        non_atomic=self.non_atomic) \
                .run(self._conn)

        else:
            reply = r.table(self.table) \
                .insert(self._data,
                        durability=self.durability,
                        upsert=self.upsert) \
                .run(self._conn)
            self._new = False

        if "generated_keys" in reply and reply["generated_keys"]:
            self._data[self.primaryKey] = reply["generated_keys"][0]

        if "errors" in reply and reply["errors"] > 0:
            raise Exception("Could not insert entry: %s"
                            % reply["first_error"])

        return True