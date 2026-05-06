def indent(indent_str=None):
    """
    A complete, standalone indent ruleset.

    Arguments:

    indent_str
        The string used for indentation.  Defaults to None, which will
        defer the value used to the one provided by the Dispatcher.
    """

    def indentation_rule():
        inst = Indentator(indent_str)
        return {'layout_handlers': {
            OpenBlock: layout_handler_openbrace,
            CloseBlock: layout_handler_closebrace,
            EndStatement: layout_handler_semicolon,
            Space: layout_handler_space_imply,
            OptionalSpace: layout_handler_space_optional_pretty,
            RequiredSpace: layout_handler_space_imply,
            Indent: inst.layout_handler_indent,
            Dedent: inst.layout_handler_dedent,
            Newline: inst.layout_handler_newline,
            OptionalNewline: inst.layout_handler_newline_optional,
            (Space, OpenBlock): NotImplemented,
            (Space, EndStatement): layout_handler_semicolon,
            (OptionalSpace, EndStatement): layout_handler_semicolon,
            (Indent, Newline, Dedent): rule_handler_noop,
        }}
    return indentation_rule