def indent(text, branch_method, leaf_method, pass_syntax, flush_left_syntax, flush_left_empty_line, indentation_method,
           get_block=get_indented_block):
    """Returns HTML as a basestring.

    Parameters
    ----------

      text : basestring
        Source code, typically SHPAML, but could be a different (but
        related) language. The remaining parameters specify details
        about the language used in the source code. To parse SHPAML,
        pass the same values as convert_shpaml_tree.

      branch_method : function
        convert_shpaml_tree passes html_block_tag here.
      leaf_method : function
        convert_shpaml_tree passes convert_line here.

      pass_syntax : basestring
        convert_shpaml_tree passes PASS_SYNTAX here.
      flush_left_syntax : basestring
        convert_shpaml_tree passes FLUSH_LEFT_SYNTAX here.
      flush_left_empty_line : basestring
        convert_shpaml_tree passes FLUSH_LEFT_EMPTY_LINE here.

      indentation_method : function
        convert_shpaml_tree passes _indent here.

      get_block : function
        Defaults to get_indented_block.
    """
    text = text.rstrip()
    lines = text.split('\n')
    if lines and lines[0].startswith('!! '):
        lines[0] = lines[0].replace('!! ', '<!DOCTYPE ') + '>'
    output = []
    indent_lines(lines, output, branch_method, leaf_method, pass_syntax, flush_left_syntax, flush_left_empty_line,
                 indentation_method, get_block=get_indented_block)
    return '\n'.join(output) + '\n'