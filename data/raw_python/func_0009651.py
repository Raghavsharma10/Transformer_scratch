def set_next_page_params(self):
        """Set the params so that the next page is fetched."""
        if self.items:
            index = self.get_last_item_index()
            self.params[self.mode] = self.get_next_page_param(self.items[index])