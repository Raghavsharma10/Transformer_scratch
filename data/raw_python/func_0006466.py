def apply_update(self, doc, update_spec):
        """Override DocManagerBase.apply_update to have flat documents."""
        # Replace a whole document
        if not '$set' in update_spec and not '$unset' in update_spec:
            # update_spec contains the new document.
            # Update the key in Solr based on the unique_key mentioned as
            # parameter.
            update_spec['_id'] = doc[self.unique_key]
            return update_spec
        for to_set in update_spec.get("$set", []):
            value = update_spec['$set'][to_set]
            # Find dotted-path to the value, remove that key from doc, then
            # put value at key:
            keys_to_pop = []
            for key in doc:
                if key.startswith(to_set):
                    if key == to_set or key[len(to_set)] == '.':
                        keys_to_pop.append(key)
            for key in keys_to_pop:
                doc.pop(key)
            doc[to_set] = value
        for to_unset in update_spec.get("$unset", []):
            # MongoDB < 2.5.2 reports $unset for fields that don't exist within
            # the document being updated.
            keys_to_pop = []
            for key in doc:
                if key.startswith(to_unset):
                    if key == to_unset or key[len(to_unset)] == '.':
                        keys_to_pop.append(key)
            for key in keys_to_pop:
                doc.pop(key)
        return doc