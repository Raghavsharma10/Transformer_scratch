def do_index_command(self, index, **options):
        """Delete search index."""
        if options["interactive"]:
            logger.warning("This will permanently delete the index '%s'.", index)
            if not self._confirm_action():
                logger.warning(
                    "Aborting deletion of index '%s' at user's request.", index
                )
                return
        return delete_index(index)