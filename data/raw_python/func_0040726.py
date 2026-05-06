def indent_lines(lines, output, branch_method, leaf_method, pass_syntax, flush_left_syntax, flush_left_empty_line,
                 indentation_method, get_block):
    """Returns None.

    The way this function produces output is by adding strings to the
    list that's passed in as the second parameter.

    Parameters
    ----------

      lines : list of basestring's
        Each string is a line of a SHPAML source code
        (trailing newlines not included).
      output : empty list
        Explained earlier...

    The remaining parameters are exactly the same as in the indent
    function:

      * branch_method
      * leaf_method
      * pass_syntax
      * flush_left_syntax
      * flush_left_empty_line
      * indentation_method
      * get_block
    """
    append = output.append

    def recurse(prefix_lines):
        while prefix_lines:
            prefix, line = prefix_lines[0]
            if line == '':
                prefix_lines.pop(0)
                append('')
                continue

            block_size = get_block(prefix_lines)
            if block_size == 1:
                prefix_lines.pop(0)
                if line == pass_syntax:
                    pass
                elif line.startswith(flush_left_syntax):
                    append(line[len(flush_left_syntax):])
                elif line.startswith(flush_left_empty_line):
                    append('')
                else:
                    append(prefix + leaf_method(line))
            else:
                block = prefix_lines[:block_size]
                prefix_lines = prefix_lines[block_size:]
                branch_method(output, block, recurse)
        return
    prefix_lines = list(map(indentation_method, lines))
    recurse(prefix_lines)