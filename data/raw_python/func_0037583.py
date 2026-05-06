def update(self, id, newObj):
        """Update a object
            Args:
                id (int): Target Object ID
                newObj (object): New object will be merged into original object
            Returns:
                Object: Updated object
                None: If specified object id is not found
                MultipleInvalid: If input object is invaild
        """
        newObj = self.validation(newObj)
        for obj in self.model.db:
            if obj["id"] != id:
                continue

            newObj.pop("id", None)
            obj.update(newObj)
            obj = self._cast_model(obj)
            if not self._batch.enable.is_set():
                self.model.save_db()
            return obj

        return None