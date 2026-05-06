def BackAssign(cls,
                   other_entity_klass,
                   this_entity_backpopulate_field,
                   other_entity_backpopulate_field,
                   is_many_to_one=False):
        """
        Assign defined one side mapping relationship to other side.

        For example, each employee belongs to one department, then one department
        includes many employees. If you defined each employee's department,
        this method will assign employees to ``Department.employees`` field.
        This is an one to many (department to employee) example.

        Another example would be, each employee has multiple tags. If you defined
        tags for each employee, this method will assign employees to
        ``Tag.employees`` field. This is and many to many (employee to tag) example.

        Support:

        - many to many mapping
        - one to many mapping

        :param other_entity_klass: a :class:`Constant` class.
        :param this_entity_backpopulate_field: str
        :param other_entity_backpopulate_field: str
        :param is_many_to_one: bool
        :return:
        """
        data = dict()
        for _, other_klass in other_entity_klass.Subclasses():
            other_field_value = getattr(
                other_klass, this_entity_backpopulate_field)
            if isinstance(other_field_value, (tuple, list)):
                for self_klass in other_field_value:
                    self_key = self_klass.__name__
                    try:
                        data[self_key].append(other_klass)
                    except KeyError:
                        data[self_key] = [other_klass, ]
            else:
                if other_field_value is not None:
                    self_klass = other_field_value
                    self_key = self_klass.__name__
                    try:
                        data[self_key].append(other_klass)
                    except KeyError:
                        data[self_key] = [other_klass, ]

        if is_many_to_one:
            new_data = dict()
            for key, value in data.items():
                try:
                    new_data[key] = value[0]
                except:  # pragma: no cover
                    pass
            data = new_data

        for self_key, other_klass_list in data.items():
            setattr(getattr(cls, self_key),
                    other_entity_backpopulate_field, other_klass_list)