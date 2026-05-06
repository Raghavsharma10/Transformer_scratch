def resolve_object_number(self, ref):
        """Resolve a variety of object numebrs to a dataset number"""

        if not isinstance(ref, ObjectNumber):
            on = ObjectNumber.parse(ref)
        else:
            on = ref

        ds_on = on.as_dataset

        return ds_on