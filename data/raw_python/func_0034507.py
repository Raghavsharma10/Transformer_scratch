def update_entries(entries: Entries, data: dict) -> None:
        """Update each entry in the list with some data."""
        # TODO: Is mutating the list okay, making copies is such a pain in the ass
        for entry in entries:
            entry.update(data)