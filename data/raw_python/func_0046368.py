def are_items_shuffled(self):
        """Tests if the items or parts appear in a random order.

        return: (boolean) - ``true`` if the items appear in a random
                order, ``false`` otherwise
        *compliance: mandatory -- This method must be implemented.*

        """
        if self._my_map['itemsShuffled'] is None:
            return self.get_assessment().are_items_shuffled()
        return bool(self._my_map['itemsShuffled'])