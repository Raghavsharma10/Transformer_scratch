def _clone_reverses(self, old_reverses):
        """
        Clones all the objects that were previously gathered.
        """

        for ctype, reverses in old_reverses.items():
            for parts in reverses.values():
                sub_objs = parts[1]
                field_name = parts[0]

                attrs = {}
                for sub_obj in sub_objs:
                    if ctype != 'm2m' and not attrs:
                        field = sub_obj._meta.get_field(field_name)
                        attrs = {
                            field.column: getattr(self, field.rel.field_name)
                        }
                    sub_obj._clone(**attrs)

                if ctype == 'm2m':
                    setattr(self, field_name, sub_objs)