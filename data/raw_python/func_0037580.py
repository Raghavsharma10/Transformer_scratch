def get(self, id):
        """Get a object by id
            Args:
                id (int): Object id

            Returns:
                Object: Object with specified id
                None: If object not found
        """
        for obj in self.model.db:
            if obj["id"] == id:
                return self._cast_model(obj)

        return None