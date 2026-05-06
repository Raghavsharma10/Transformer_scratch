def get_or_create_placeholder(self, page, placeholder_slot):
        """
        Add a placeholder if not exists.
        """
        placeholder, created = get_or_create_placeholder(
            page, placeholder_slot, delete_existing=self.delete_first)
        return placeholder, created