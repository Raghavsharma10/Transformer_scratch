def itervisit_node(self, node):
        '''Given a node, find the matching visitor function (if any) and
        run it. If the result is a context manager, yield from all the nodes
        children before allowing it to exit. Otherwise, return the result.
        '''
        # Get the corresponding method and run it.
        func = self.get_method(node)
        if func is None:
            generic_visit = getattr(self, 'generic_visit', None)
            if generic_visit is not None:
                result = generic_visit(node)
            else:
                # There is no handler defined for this node.
                return
        else:
            result = self.apply_visitor_method(func, node)

        # If result is a generator, yield from it.
        if isinstance(result, self.GeneratorType):
            yield from result

        # If result is a context manager, enter, visit children, then exit.
        elif isinstance(result, self.GeneratorContextManager):
            with result:
                itervisit_nodes = self.itervisit_nodes
                for child in self.get_children(node):
                    try:
                        yield from itervisit_nodes(child)
                    except self.Continue:
                        continue

        # Otherwise just yield the result.
        else:
            yield result