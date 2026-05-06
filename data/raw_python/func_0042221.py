def _convert_value(self, item):
        """
        Handle different value types for XLS. Item is a cell object.
        """
        # Types:
        # 0 = empty u''
        # 1 = unicode text
        # 2 = float (convert to int if possible, then convert to string)
        # 3 = date (convert to unambiguous date/time string)
        # 4 = boolean (convert to string "0" or "1")
        # 5 = error (convert from code to error text)
        # 6 = blank u''

        # Thx to Augusto C Men to point fast solution for XLS/XLSX dates
        if item.ctype == 3:  # XL_CELL_DATE:
            try:
                return datetime.datetime(*xlrd.xldate_as_tuple(item.value, self._book.datemode))
            except ValueError:
                # TODO: make toggable
                # Invalid date
                return item.value

        if item.ctype == 2:  # XL_CELL_NUMBER:
            if item.value % 1 == 0:  # integers
                return int(item.value)
            else:
                return item.value

        return item.value