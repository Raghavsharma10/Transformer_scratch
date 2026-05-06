def _copy(self):
        """ needs to update page numbers """
        ins = copy.copy(self)
        ins._fire_page_number(self.page_number + 1)
        return ins