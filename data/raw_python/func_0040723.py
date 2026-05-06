def text(short):
    """Compiles short markup text into an HTML strings"""
    return indent(short, branch_method=html_block_tag, leaf_method=convert_line, pass_syntax=PASS_SYNTAX,
                  flush_left_syntax=FLUSH_LEFT_SYNTAX, flush_left_empty_line=FLUSH_LEFT_EMPTY_LINE,
                  indentation_method=find_indentation)