def is_valid(self, instance):
        '''Perform validation for *instance* and stores serialized data,
indexes and errors into local cache.
Return ``True`` if the instance is ready to be saved to database.'''
        dbdata = instance.dbdata
        data = dbdata['cleaned_data'] = {}
        errors = dbdata['errors'] = {}
        #Loop over scalar fields first
        for field, value in instance.fieldvalue_pairs():
            name = field.attname
            try:
                svalue = field.set_get_value(instance, value)
            except Exception as e:
                errors[name] = str(e)
            else:
                if (svalue is None or svalue is '') and field.required:
                    errors[name] = ("Field '{0}' is required for '{1}'."
                                    .format(name, self))
                else:
                    if isinstance(svalue, dict):
                        data.update(svalue)
                    elif svalue is not None:
                        data[name] = svalue
        return len(errors) == 0