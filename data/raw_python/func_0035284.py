def pretty_print(ast, indent_str='  '):
    """
    Simple pretty print function; returns a string rendering of an input
    AST of an ES5 Program.

    arguments

    ast
        The AST to pretty print
    indent_str
        The string used for indentations.  Defaults to two spaces.
    """

    return ''.join(chunk.text for chunk in pretty_printer(indent_str)(ast))