def _update_data(self, data={}):
        '''Update the data in this object.'''

        # Store the changes to prevent this update from affecting it
        pending_changes = self._changes or {}
        try:
            del self._changes
        except:
            pass

        # Map custom fields into our custom fields object
        try:
            custom_field_data = data.pop('custom_fields')
        except KeyError:
            pass
        else:
            self.custom_fields = Custom_Fields(custom_field_data)

        # Map all other dictionary data to object attributes
        for key, value in data.iteritems():
            lookup_key = self._field_type.get(key, key)

            # if it's a datetime object, turn into proper DT object
            if lookup_key == 'datetime' or lookup_key == 'date':
                self.__dict__[key] = datetime_parse(value)
            else:
                # Check to see if there's cache data for this item.
                # Will return an object if it's recognized as one.
                self.__dict__[key] = self._redmine.check_cache(lookup_key, value)


        # Set the changes dict to track all changes from here on out
        self._changes = pending_changes