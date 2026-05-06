def _add_text(self, text):
        """
        If ``text`` is not empty, append a new Text node to the most recent
        pending node, if there is any, or to the new nodes, if there are no
        pending nodes.
        """
        if text:
            if self.pending_nodes:
                self.pending_nodes[-1].append(nodes.Text(text))
            else:
                self.new_nodes.append(nodes.Text(text))