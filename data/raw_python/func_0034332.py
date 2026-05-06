def add_list(self, bl):
        """
        note that this pushes the cur_element, but doesn't pop it.
        You'll need to do that
        """
        # text:list doesn't like being a child of text:p
        if self.cur_element is None:
            self.add_text_frame()
        self.push_element()
        self.cur_element._text_box.append(bl.node)
        style = bl.style_name
        if style not in self._preso._styles_added:
            self._preso._styles_added[style] = 1
            content = bl.default_styles_root()[0]
            self._preso._auto_styles.append(content)
        self.cur_element = bl