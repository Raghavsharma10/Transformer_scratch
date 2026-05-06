def add(self, obj):
        """Add a object
            Args:
                Object: Object will be added
            Returns:
                Object: Object with id
            Raises:
                TypeError: If add object is not a dict
                MultipleInvalid: If input object is invaild
        """
        if not isinstance(obj, dict):
            raise TypeError("Add object should be a dict object")
        obj = self.validation(obj)
        obj["id"] = self.maxId + 1
        obj = self._cast_model(obj)
        self.model.db.append(obj)

        if not self._batch.enable.is_set():
            self.model.save_db()
        return obj