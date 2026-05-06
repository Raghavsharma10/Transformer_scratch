def are_items_sequential(self):
        """Tests if the items or parts in this assessment are taken sequentially.

        return: (boolean) - ``true`` if the items are taken
                sequentially, ``false`` if the items can be skipped and
                revisited
        *compliance: mandatory -- This method must be implemented.*

        """
        if self._my_map['itemsSequential'] is None:
            return self.get_assessment().are_items_sequential()
        return bool(self._my_map['itemsSequential'])