def stylize_comment_block(lines):
  """Parse comment lines and make subsequent indented lines into a code block
  block.
  """
  normal, sep, in_code = range(3)
  state = normal
  for line in lines:
    indented = line.startswith('    ')
    empty_line = line.strip() == ''

    if state == normal and empty_line:
      state = sep
    elif state in [sep, normal] and indented:
      yield ''
      if indented:
        yield '.. code-block:: javascript'
        yield ''
        yield line
        state = in_code
      else:
        state = normal
    elif state == sep and not empty_line:
      yield ''
      yield line
      state = normal
    else:
      yield line
      if state == in_code and not (indented or empty_line):
        sep = normal