def _process_getitem(self, node, **kwargs):
        """
        Processes a `GetItem` node e.g. {{ foo["bar"] }}
        """

        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self._process_node(node.node, **new_kwargs)

                if isinstance(node.arg, nodes.Slice):
                    self.output.write('.slice(')

                    if node.arg.step is not None:
                        raise Exception('The step argument is not supported when slicing.')

                    if node.arg.start is None:
                        self.output.write('0')
                    else:
                        self._process_node(node.arg.start, **new_kwargs)

                    if node.arg.stop is None:
                        self.output.write(')')
                    else:
                        self.output.write(',')
                        self._process_node(node.arg.stop, **new_kwargs)
                        self.output.write(')')
                else:
                    self.output.write('[')
                    self._process_node(node.arg, **new_kwargs)
                    self.output.write(']')