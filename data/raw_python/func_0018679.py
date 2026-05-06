def _restrict_items_to_count(items: List[Tuple[Key, Any]], count: int) -> List[Tuple[Key, Any]]:
        """
        Restricts items to count number if len(items) is larger than abs(count). This function
        assumes that items is sorted by time.

        :param items: The items to restrict.
        :param count: The number of items returned.
        """
        if abs(count) > len(items):
            count = Store._sign(count) * len(items)

        if count < 0:
            return items[count:]
        else:
            return items[:count]