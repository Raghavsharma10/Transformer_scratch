def _get_indexes_by_path(self, field):
        """
        Returns a list of indexes by field path.

        :param field: Field structure as following:
         *.subfield_2  would apply the function to the every subfield_2 of the elements
         1.subfield_2  would apply the function to the subfield_2 of the element 1
         * would apply the function to every element
         1 would apply the function to element 1
        """
        try:
            field, next_field = field.split('.', 1)
        except ValueError:
            next_field = ''

        if field == '*':
            index_list = []
            for item in self:
                index_list.append(self.index(item))
            if index_list:
                return index_list, next_field
            return [], None
        elif field.isnumeric():
            index = int(field)
            if index >= len(self):
                return None, None
            return [index], next_field