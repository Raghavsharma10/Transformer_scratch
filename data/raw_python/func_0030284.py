def get_parent_page(self):
        """
        For 'parent' in cms.api.create_page()
        """
        if self.current_level == 1:
            # 'root' page
            return None
        else:
            return self.page_data[(self.current_level - 1, self.current_count)]