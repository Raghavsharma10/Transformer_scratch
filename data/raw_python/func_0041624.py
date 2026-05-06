def _struct_to_values(self, structure, source_code):
        """Return list of values readed in source_code, 
        according to given structure.
        """
        # iterate on source code until all values are finded
        # if a value is not foundable, 
        #   (ie its identificator is not in source code)
        #   it will be replaced by associated neutral value
        iter_source_code = itertools.cycle(source_code)
        values = []
        for lexem_type in (l for l in structure if l is not 'D'):
            if lexem_type is LEXEM_TYPE_CONDITION:
                new_value = self._next_condition_lexems(
                    iter_source_code, len(source_code)
                )
            else:
                new_value = self._next_lexem(
                    lexem_type, iter_source_code, len(source_code)
                )
            # if values is unvalid:
            #   association with the right neutral value
            if new_value is None:
                if lexem_type in (LEXEM_TYPE_PREDICAT, LEXEM_TYPE_CONDITION):
                    new_value = self.neutral_value_condition
                else:
                    new_value = self.neutral_value_action
            values.append(new_value)

        return values