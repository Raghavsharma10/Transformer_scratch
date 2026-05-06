def indent(indent_str=None):
    """
    An example indentation ruleset.
    """

    def indentation_rule():
        inst = Indentator(indent_str)
        return {'layout_handlers': {
            Indent: inst.layout_handler_indent,
            Dedent: inst.layout_handler_dedent,
            Newline: inst.layout_handler_newline,
            OptionalNewline: inst.layout_handler_newline_optional,
            OpenBlock: layout_handler_openbrace,
            CloseBlock: layout_handler_closebrace,
            EndStatement: layout_handler_semicolon,
        }}
    return indentation_rule