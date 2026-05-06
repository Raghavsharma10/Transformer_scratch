def push_pending_node(self, name, attr):
        """
        pending nodes are for affecting type, such as wrapping content
        with text:a to make a hyperlink.  Anything in pending nodes
        will be written before the actual text.
        User needs to remember to pop out of it.
        """
        if self.cur_element is None:
            self.add_text_frame()
        self.cur_element.pending_nodes.append((name, attr))