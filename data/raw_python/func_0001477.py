def inspect_decorator(lines: List[str], lineno: int, filename: str) -> DecoratorInspection:
    """
    Parse the file in which the decorator is called and figure out the corresponding call AST node.

    :param lines: lines of the source file corresponding to the decorator call
    :param lineno: line index (starting with 0) of one of the lines in the decorator call
    :param filename: name of the file where decorator is called
    :return: inspected decorator call
    """
    if lineno < 0 or lineno >= len(lines):
        raise ValueError(("Given line number {} of one of the decorator lines "
                          "is not within the range [{}, {}) of lines in {}").format(lineno, 0, len(lines), filename))

    # Go up till a line starts with a decorator
    decorator_lineno = None  # type: Optional[int]
    for i in range(lineno, -1, -1):
        if _DECORATOR_RE.match(lines[i]):
            decorator_lineno = i
            break

    if decorator_lineno is None:
        raise SyntaxError("Decorator corresponding to the line {} could not be found in file {}: {!r}".format(
            lineno + 1, filename, lines[lineno]))

    # Find the decorator end -- it's either a function definition, a class definition or another decorator
    decorator_end_lineno = None  # type: Optional[int]
    for i in range(lineno + 1, len(lines)):
        line = lines[i]

        if _DECORATOR_RE.match(line) or _DEF_CLASS_RE.match(line):
            decorator_end_lineno = i
            break

    if decorator_end_lineno is None:
        raise SyntaxError(("The next statement following the decorator corresponding to the line {} "
                           "could not be found in file {}: {!r}").format(lineno + 1, filename, lines[lineno]))

    decorator_lines = lines[decorator_lineno:decorator_end_lineno]

    # We need to dedent the decorator and add a dummy decoratee so that we can parse its text as valid source code.
    decorator_text = textwrap.dedent("".join(decorator_lines)) + "def dummy_{}(): pass".format(uuid.uuid4().hex)

    atok = asttokens.ASTTokens(decorator_text, parse=True)

    assert isinstance(atok.tree, ast.Module), "Expected the parsed decorator text to live in an AST module."

    module_node = atok.tree
    assert len(module_node.body) == 1, "Expected the module AST of the decorator text to have a single statement."
    assert isinstance(module_node.body[0], ast.FunctionDef), \
        "Expected the only statement in the AST module corresponding to the decorator text to be a function definition."

    func_def_node = module_node.body[0]

    assert len(func_def_node.decorator_list) == 1, \
        "Expected the function AST node corresponding to the decorator text to have a single decorator."

    assert isinstance(func_def_node.decorator_list[0], ast.Call), \
        "Expected the only decorator in the function definition AST node corresponding to the decorator text " \
        "to be a call node."

    call_node = func_def_node.decorator_list[0]

    return DecoratorInspection(atok=atok, node=call_node)