def generic_visit(self, node, iter_fields=ast.iter_fields, AST=ast.AST):
        """Called if no explicit visitor function exists for a node.
        """
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        self.visit(item)
            elif isinstance(value, AST):
                self.visit(value)