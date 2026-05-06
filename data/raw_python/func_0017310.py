def newick_from_string(self):
        "Reads a newick string in the New Hampshire format."

        # split on parentheses to traverse hierarchical tree structure
        for chunk in self.data.split("(")[1:]:
            # add child to make this node a parent.
            self.current_parent = (
                self.root if self.current_parent is None else
                self.current_parent.add_child()
            )

            # get all parenth endings from this parenth start
            subchunks = [ch.strip() for ch in chunk.split(",")]
            if subchunks[-1] != '' and not subchunks[-1].endswith(';'):
                raise NewickError(
                    'Broken newick structure at: {}'.format(chunk))

            # Every closing parenthesis will close a node and go up one level.
            for idx, leaf in enumerate(subchunks):
                if leaf.strip() == '' and idx == len(subchunks) - 1:
                    continue
                closing_nodes = leaf.split(")")

                # parse features and apply to the node object
                self.apply_node_data(closing_nodes[0], "leaf")

                # next contain closing nodes and data about the internal nodes.
                if len(closing_nodes) > 1:
                    for closing_internal in closing_nodes[1:]:
                        closing_internal = closing_internal.rstrip(";")
                        # read internal node data and go up one level
                        self.apply_node_data(closing_internal, "internal")
                        self.current_parent = self.current_parent.up
        return self.root