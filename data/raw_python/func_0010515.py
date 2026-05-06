def _scoped_variables(self, nodes_list, **kwargs):
        """
        Context manager for creating scoped variables defined by the nodes in `nodes_list`.
        These variables will be added to the context, and when the context manager exits the
        context object will be restored to it's previous state.
        """

        tmp_vars = []
        for node in nodes_list:

            is_assign_node = isinstance(node, nodes.Assign)
            name = node.target.name if is_assign_node else node.name

            # create a temp variable name
            tmp_var = next(self.temp_var_names)

            # save previous context value
            with self._execution():

                # save the current value of this name
                self.output.write('var %s = %s.%s;' % (tmp_var, self.context_name, name))

                # add new value to context
                self.output.write('%s.%s = ' % (self.context_name, name))

                if is_assign_node:
                    self._process_node(node.node, **kwargs)
                else:
                    self.output.write(node.name)

                self.output.write(';')

            tmp_vars.append((tmp_var, name))

        yield

        # restore context
        for tmp_var, name in tmp_vars:
            with self._execution():
                self.output.write('%s.%s = %s;' % (self.context_name, name, tmp_var))