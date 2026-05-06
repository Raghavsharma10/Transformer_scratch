def _process_math(self, node, math_operator=None, function=None, **kwargs):
        """
        Processes a math node e.g. `Div`, `Sub`, `Add`, `Mul` etc...
        If `function` is provided the expression is wrapped in a call to that function.
        """

        with self._interpolation():
            if function:
                self.output.write(function)
                self.output.write('(')

            self._process_node(node.left, **kwargs)
            self.output.write(math_operator)
            self._process_node(node.right, **kwargs)

            if function:
                self.output.write(')')