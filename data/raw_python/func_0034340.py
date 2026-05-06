def line_break(self):
        """insert as many line breaks as the insert_line_break variable says
        """
        for i in range(self.slide.insert_line_break):
            # needs to be inside text:p
            if not self._in_tag(ns("text", "p")):
                # we can just add a text:p and no line-break
                # Create paragraph style first
                self.add_node(ns("text", "p"))
            self.add_node(ns("text", "line-break"))
            self.pop_node()
            if self.cur_node.tag == ns("text", "p"):
                return

            if self.cur_node.getparent().tag != ns("text", "p"):
                self.pop_node()
        self.slide.insert_line_break = 0