def set(self, id, newObj):
        """Set a object
            Args:
                id (int): Target Object ID
                newObj (object): New object will be set
            Returns:
                Object: New object
                None: If specified object id is not found
                MultipleInvalid: If input object is invaild
        """
        newObj = self.validation(newObj)
        for index in xrange(0, len(self.model.db)):
            if self.model.db[index]["id"] != id:
                continue

            newObj["id"] = id
            self.model.db[index] = self._cast_model(newObj)
            if not self._batch.enable.is_set():
                self.model.save_db()
            return self.model.db[index]

        return None