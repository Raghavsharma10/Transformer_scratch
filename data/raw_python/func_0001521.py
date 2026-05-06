def _execute_comprehension(self, node: Union[ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp]) -> Any:
        """Compile the generator or comprehension from the node and execute the compiled code."""
        args = [ast.arg(arg=name) for name in sorted(self._name_to_value.keys())]

        func_def_node = ast.FunctionDef(
            name="generator_expr",
            args=ast.arguments(args=args, kwonlyargs=[], kw_defaults=[], defaults=[]),
            decorator_list=[],
            body=[ast.Return(node)])

        module_node = ast.Module(body=[func_def_node])

        ast.fix_missing_locations(module_node)

        code = compile(source=module_node, filename='<ast>', mode='exec')

        module_locals = {}  # type: Dict[str, Any]
        module_globals = {}  # type: Dict[str, Any]
        exec(code, module_globals, module_locals)  # pylint: disable=exec-used

        generator_expr_func = module_locals["generator_expr"]

        return generator_expr_func(**self._name_to_value)