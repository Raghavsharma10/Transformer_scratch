def _process_getattr(self, node, **kwargs):
        """
        Processes a `GetAttr` node. e.g. {{ foo.bar }}
        """

        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                if is_loop_helper(node):
                    self._process_loop_helper(node, **new_kwargs)
                else:
                    self._process_node(node.node, **new_kwargs)
                    self.output.write('.')
                    self.output.write(node.attr)