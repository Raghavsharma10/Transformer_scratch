def _process_loop_helper(self, node, **kwargs):
        """
        Processes a loop helper e.g. {{ loop.first }} or {{ loop.index }}
        """

        if node.attr == LOOP_HELPER_INDEX:
            self.output.write('(arguments[1] + 1)')
        elif node.attr == LOOP_HELPER_INDEX_0:
            self.output.write('arguments[1]')
        elif node.attr == LOOP_HELPER_FIRST:
            self.output.write('(arguments[1] == 0)')
        elif node.attr == LOOP_HELPER_LAST:
            self.output.write('(arguments[1] == arguments[2].length - 1)')
        elif node.attr == LOOP_HELPER_LENGTH:
            self.output.write('arguments[2].length')