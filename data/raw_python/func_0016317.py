def do_index_command(self, index, **options):
        """Rebuild search index."""
        if options["interactive"]:
            logger.warning("This will permanently delete the index '%s'.", index)
            if not self._confirm_action():
                logger.warning(
                    "Aborting rebuild of index '%s' at user's request.", index
                )
                return

        try:
            delete = delete_index(index)
        except TransportError:
            delete = {}
            logger.info("Index %s does not exist, cannot be deleted.", index)
        create = create_index(index)
        update = update_index(index)

        return {"delete": delete, "create": create, "update": update}