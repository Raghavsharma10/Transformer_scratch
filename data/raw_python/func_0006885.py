def letter_set(self):
        """
        Return the letter set of this node.
        """
        end_str = ctypes.create_string_buffer(MAX_CHARS)

        cgaddag.gdg_letter_set(self.gdg, self.node, end_str)

        return [char for char in end_str.value.decode("ascii")]