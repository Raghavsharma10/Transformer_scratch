def _process_if(self, node, execute_end=None, **kwargs):
        """
        Processes an if block e.g. `{% if foo %} do something {% endif %}`
        """

        with self._execution():
            self.output.write('if')
            self.output.write('(')

            with option(kwargs, use_python_bool_wrapper=True):
                self._process_node(node.test, **kwargs)

            self.output.write(')')
            self.output.write('{')

        # We accept an `execute_end` function as a keyword argument as this function is
        # recursive in the case of something like if-elif-elif-else. In these cases this
        # invocation of this function may have to close execution opened by a previous
        # invocation of this function.
        if execute_end:
            execute_end()

        # body
        for n in node.body:
            self._process_node(n, **kwargs)

        if not node.else_ and not node.elif_:
            # no else - just close the if
            with self._execution():
                self.output.write('}')

        else:
            # either an else or an elif
            with self._execution() as execute_end:
                self.output.write('}')
                self.output.write(' else ')

                # check for elif
                for n in node.elif_:
                    self._process_node(n, execute_end=execute_end, **kwargs)

                if node.elif_ and node.else_:
                    self.output.write(' else ')

                # open up the body
                self.output.write('{')

            # process the body of the else
            for n in node.else_:
                self._process_node(n, **kwargs)

            # close the body
            with self._execution():
                self.output.write('}')