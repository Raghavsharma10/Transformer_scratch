def edges(self):
        """
        Return the edge characters of this node.
        """
        edge_str = ctypes.create_string_buffer(MAX_CHARS)

        cgaddag.gdg_edges(self.gdg, self.node, edge_str)

        return [char for char in edge_str.value.decode("ascii")]