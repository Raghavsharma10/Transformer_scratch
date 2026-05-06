def get_ordering_field_columns(self):
        """
        Returns an OrderedDict of ordering field column numbers and asc/desc
        """

        # We must cope with more than one column having the same underlying
        # sort field, so we base things on column numbers.
        ordering = self._get_default_ordering()
        ordering_fields = OrderedDict()
        if ORDER_VAR not in self.params:
            # for ordering specified on model_admin or model Meta, we don't
            # know the right column numbers absolutely, because there might be
            # morr than one column associated with that ordering, so we guess.
            for field in ordering:
                if field.startswith('-'):
                    field = field[1:]
                    order_type = 'desc'
                else:
                    order_type = 'asc'
                for index, attr in enumerate(self.list_display):
                    if self.get_ordering_field(attr) == field:
                        ordering_fields[index] = order_type
                        break
        else:
            for p in self.params[ORDER_VAR].split('.'):
                none, pfx, idx = p.rpartition('-')
                try:
                    idx = int(idx)
                except ValueError:
                    continue  # skip it
                ordering_fields[idx] = 'desc' if pfx == '-' else 'asc'
        return ordering_fields