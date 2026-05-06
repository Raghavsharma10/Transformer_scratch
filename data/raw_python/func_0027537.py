def add_instruction(self, target, data):
        """
        Add an instruction node to this element.

        :param string text: text content to add as an instruction.
        """
        self._add_instruction(self.impl_node, target, data)