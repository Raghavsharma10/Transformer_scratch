def remove_condition(self, manager, condition_set, field_name, condition,
                         commit=True):
        """
        Removes a condition and updates the global ``operator`` switch manager.

        If ``commit`` is ``False``, the data will not be written to the
        database.

        >>> switch = operator['my_switch'] #doctest: +SKIP
        >>> cs_id = condition_set.get_id() #doctest: +SKIP
        >>> switch.remove_condition(cs_id, 'percent', [0, 50]) #doctest: +SKIP
        """
        condition_set = manager.get_condition_set_by_id(condition_set)

        namespace = condition_set.get_namespace()

        if namespace not in self.value:
            return

        if field_name not in self.value[namespace]:
            return

        conditions = self.value[namespace][field_name]
        self.value[namespace][field_name] = ([c for c in conditions
                                             if c[1] != condition])

        if not self.value[namespace][field_name]:
            del self.value[namespace][field_name]

            if not self.value[namespace]:
                del self.value[namespace]

        if commit:
            self.save()