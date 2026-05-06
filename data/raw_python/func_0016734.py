def _value_ref(self, column, value, *, dumped=False, inner=False):
        """inner=True uses column.typedef.inner_type instead of column.typedef"""
        ref = ":v{}".format(self.next_index)

        # Need to dump this value
        if not dumped:
            typedef = column.typedef
            for segment in path_of(column):
                typedef = typedef[segment]
            if inner:
                typedef = typedef.inner_typedef
            value = self.engine._dump(typedef, value)

        self.attr_values[ref] = value
        self.counts[ref] += 1
        return ref, value