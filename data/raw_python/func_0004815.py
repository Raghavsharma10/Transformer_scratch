def add_predicate(self, key, value, predicate_type='equals'):
        """
        add key, value, type combination of a predicate

        :param key: query KEY parameter
        :param value: the value used in the predicate
        :param predicate_type: the type of predicate (e.g. ``equals``)
        """
        if predicate_type not in operators:
            predicate_type = operator_lkup.get(predicate_type)
        if predicate_type:
            self.predicates.append({'type': predicate_type,
                                    'key': key,
                                    'value': value
                                    })
        else:
            raise Exception("predicate type not a valid operator")