def save(self):
        '''Save all changes on this item (if any) back to Redmine.'''
        self._check_custom_fields()

        if not self._changes:
            return None

        for tag in self._remap_to_id:
            self._remap_tag_to_tag_id(tag, self._changes)

        # Check for custom handlers for tags
        for tag, type in self._field_type.items():
            try:
                raw_data = self._changes[tag]
            except:
                continue

            # Convert datetime type to a datetime string that Redmine expects
            if type == 'datetime':
                try:
                    self._changes[tag] = raw_data.strftime('%Y-%m-%dT%H:%M:%S%z')
                except AttributeError:
                    continue

            # Convert date type to a date string that Redmine expects
            if type == 'date':
                try:
                    self._changes[tag] = raw_data.strftime('%Y-%m-%d')
                except AttributeError:
                    continue


        try:
            self._update(self._changes)
        except:
            raise
        else:
            # Successful save, woot! Now clear the changes dict
            self._changes.clear()