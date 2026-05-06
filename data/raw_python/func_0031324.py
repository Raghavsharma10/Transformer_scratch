def obj_to_dict(cls, obj):
        """
        Takes a model object and converts it into a dictionary suitable for
        passing to the constructor's data attribute.
        """
        data = {}
        for field_name in cls.get_fields():
            try:
                value = getattr(obj, field_name)
            except AttributeError:
                # If the field doesn't exist on the object, fail gracefully
                # and don't include the field in the data dict at all. Fail
                # loudly if the field exists but produces a different error
                # (edge case: accessing an *existing* field could technically
                # produce an unrelated AttributeError).
                continue

            if callable(value):
                value = value()
            data[field_name] = value

        return data