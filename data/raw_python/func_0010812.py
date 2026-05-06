def lookup_obj_attribute(self, obj, field):
        """
        Looks for a field's value from the passed in obj.  Note that this will strip
        leading attributes to deal with subelements if possible
        """
        curr_field = field.encode('ascii', 'ignore').decode("utf-8")
        rest = None

        if field.find('.') >= 0:
            curr_field = field.split('.')[0]
            rest = '.'.join(field.split('.')[1:])

        # next up is the object itself
        obj_field = getattr(obj, curr_field, None)

        # if it is callable, do so
        if obj_field and getattr(obj_field, '__call__', None):
            obj_field = obj_field()

        if obj_field and rest:
            return self.lookup_obj_attribute(obj_field, rest)
        else:
            return obj_field