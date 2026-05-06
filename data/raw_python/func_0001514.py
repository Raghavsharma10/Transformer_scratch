def visit_Call(self, node: ast.Call) -> Any:
        """Visit the function and the arguments and finally make the function call with them."""
        func = self.visit(node=node.func)

        args = []  # type: List[Any]
        for arg_node in node.args:
            if isinstance(arg_node, ast.Starred):
                args.extend(self.visit(node=arg_node))
            else:
                args.append(self.visit(node=arg_node))

        kwargs = dict()  # type: Dict[str, Any]
        for keyword in node.keywords:
            if keyword.arg is None:
                kw = self.visit(node=keyword.value)
                for key, val in kw.items():
                    kwargs[key] = val

            else:
                kwargs[keyword.arg] = self.visit(node=keyword.value)

        result = func(*args, **kwargs)

        self.recomputed_values[node] = result
        return result