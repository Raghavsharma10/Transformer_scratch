def get_grade_ids(self):
        """Gets the grade Ids in this system ranked from highest to lowest.

        return: (osid.id.IdList) - the list of grades Ids
        raise:  IllegalState - is_based_on_grades() is false
        compliance: mandatory - This method must be implemented.

        """
        id_list = []
        for grade_map in self._my_map['grades']:
            id_list.append(Id(grade_map.id))
        if id_list == []:
            raise IllegalState()
        else:
            return IdList(id_list)