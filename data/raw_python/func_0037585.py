def getAll(self):
        """Get all objects
            Returns:
                List: list of all objects
        """
        objs = []
        for obj in self.model.db:
            objs.append(self._cast_model(obj))

        return objs