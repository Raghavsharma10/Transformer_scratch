def get_func_lno(self, funcname):
        """The first line number of the last defined 'funcname' function."""

        class FuncLineno(ast.NodeVisitor):
            def __init__(self):
                self.clss = []

            def generic_visit(self, node):
                for child in ast.iter_child_nodes(node):
                    for item in self.visit(child):
                        yield item

            def visit_ClassDef(self, node):
                self.clss.append(node.name)
                for item in self.generic_visit(node):
                    yield item
                self.clss.pop()

            def visit_FunctionDef(self, node):
                # Only allow non nested function definitions.
                name = '.'.join(itertools.chain(self.clss, [node.name]))
                yield name, node.lineno

        if self.functions_firstlno is None:
            self.functions_firstlno = {}
            for name, lineno in FuncLineno().visit(self.node):
                if (name not in self.functions_firstlno or
                        self.functions_firstlno[name] < lineno):
                    self.functions_firstlno[name] = lineno
        try:
            return self.functions_firstlno[funcname]
        except KeyError:
            raise BdbSourceError('{}: function "{}" not found.'.format(
                self.filename, funcname))