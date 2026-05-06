def get_active_conditions(self, manager):
        '''
        Returns a generator which yields groups of lists of conditions.

        >>> conditions = switch.get_active_conditions()
        >>> for label, set_id, field, value, exc in conditions: #doctest: +SKIP
        >>>     print ("%(label)s: %(field)s = %(value)s (exclude: %(exc)s)"
        >>>            % (label, field.label, value, exc)) #doctest: +SKIP
        '''
        for condition_set in sorted(manager.get_condition_sets(),
                                    key=lambda x: x.get_group_label()):
            ns = condition_set.get_namespace()
            condition_set_id = condition_set.get_id()
            if ns in self.value:
                group = condition_set.get_group_label()
                for name, field in condition_set.fields.iteritems():
                    for value in self.value[ns].get(name, []):
                        try:
                            yield (condition_set_id, group, field, value[1],
                                   value[0] == EXCLUDE)
                        except TypeError:
                            continue