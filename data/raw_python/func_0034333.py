def add_table(self, t):
        """
        remember to call pop_element after done with table
        """
        self.push_element()
        self._page.append(t.node)
        self.cur_element = t