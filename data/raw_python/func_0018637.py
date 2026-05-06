def insert(self, index: int, obj: Any) -> None:
        """ Inserts an item to the list as long as it is not None """
        if obj is not None:
            super().insert(index, obj)