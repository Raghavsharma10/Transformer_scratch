def _attach_new(self, records, current, touch=True):
        """
        Attach all of the IDs that aren't in the current dict.
        """
        changes = {
            'attached': [],
            'updated': []
        }

        for id, attributes in records.items():
            if id not in current:
                self.attach(id, attributes, touch)

                changes['attached'].append(id)
            elif len(attributes) > 0 and self.update_existing_pivot(id, attributes, touch):
                changes['updated'].append(id)

        return changes