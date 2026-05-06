def ast_imports(file_path):
    """get list of import from python module
    :return: (list - tuple) (module, name, asname, level)
    """
    with open(file_path, 'r') as fp:
        text = fp.read()
    mod_ast = ast.parse(text, file_path)
    finder = _ImportsFinder()
    finder.visit(mod_ast)
    return finder.imports