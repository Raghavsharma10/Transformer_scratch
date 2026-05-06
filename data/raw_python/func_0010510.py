def _process_for(self, node, **kwargs):
        """
        Processes a for loop. e.g.
            {% for number in numbers %}
                {{ number }}
            {% endfor %}
            {% for key, value in somemap.items() %}
                {{ key }} -> {{ value }}
            {% %}
        """

        # since a for loop can introduce new names into the context
        # we need to remember the ones that existed outside the loop
        previous_stored_names = self.stored_names.copy()

        with self._execution():
            self.output.write('__runtime.each(')

            if is_method_call(node.iter, dict.keys.__name__):
                self.output.write('Object.keys(')

            self._process_node(node.iter, **kwargs)

            if is_method_call(node.iter, dict.keys.__name__):
                self.output.write(')')

            self.output.write(',')
            self.output.write('function')
            self.output.write('(')

            # javascript iterations put the value first, then the key
            if isinstance(node.target, nodes.Tuple):
                if len(node.target.items) > 2:
                    raise Exception('De-structuring more than 2 items is not supported.')

                for i, item in enumerate(reversed(node.target.items)):
                    self._process_node(item, **kwargs)
                    if i < len(node.target.items) - 1:
                        self.output.write(',')
            else:
                self._process_node(node.target, **kwargs)

            self.output.write(')')
            self.output.write('{')

            if node.test:
                self.output.write('if (!(')
                self._process_node(node.test, **kwargs)
                self.output.write(')) { return; }')

        assigns = node.target.items if isinstance(node.target, nodes.Tuple) else [node.target]

        with self._scoped_variables(assigns, **kwargs):
            for n in node.body:
                self._process_node(n, **kwargs)

        with self._execution():
            self.output.write('}')
            self.output.write(')')
            self.output.write(';')

        # restore the stored names
        self.stored_names = previous_stored_names