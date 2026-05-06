def handle(self, client, subhooks=()):
        """Handle a new update.

        Fetches new data from the client, then compares it to the previous
        lookup.

        Returns:
            (bool, new_data): whether changes occurred, and the new value.
        """
        new_data = self.fetch(client)

        # Holds the list of updated fields.
        updated = {}

        if not subhooks:
            # We always want to compare to previous values.
            subhooks = [self.name]

        for subhook in subhooks:
            new_key = self.extract_key(new_data, subhook)
            if new_key != self.previous_keys.get(subhook):
                updated[subhook] = new_key

        if updated:
            logger.debug("Hook %s: data changed from %r to %r", self.name, self.previous_keys, updated)
            self.previous_keys.update(updated)
            return (True, new_data)

        return (False, None)