def lookup_field_value(self, context, obj, field):
        """
        Looks up the field value for the passed in object and field name.

        Note that this method is actually called from a template, but this provides a hook
        for subclasses to modify behavior if they wish to do so.

        This may be used for example to change the display value of a variable depending on
        other variables within our context.
        """
        curr_field = field.encode('ascii', 'ignore').decode("utf-8")

        # if this isn't a subfield, check the view to see if it has a get_ method
        if field.find('.') == -1:
            # view supercedes all, does it have a 'get_' method for this obj
            view_method = getattr(self, 'get_%s' % curr_field, None)
            if view_method:
                return view_method(obj)

        return self.lookup_obj_attribute(obj, field)