def _process_name(self, node, **kwargs):
        """
        Processes a `Name` node. Some examples of `Name` nodes:
            {{ foo }} -> 'foo' is a Name
            {% if foo }} -> 'foo' is a Name
        """

        with self._interpolation():
            with self._python_bool_wrapper(**kwargs):

                if node.name not in self.stored_names and node.ctx != 'store':
                    self.output.write(self.context_name)
                    self.output.write('.')

                if node.ctx == 'store':
                    self.stored_names.add(node.name)

                self.output.write(node.name)