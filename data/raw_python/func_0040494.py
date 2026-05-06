def inspect(self, tab_width=2, ident_char='-'):
    """
    Inspects a project file structure based based on the instance folder property.

    :param tab_width: width size for subfolders and files.
    :param ident_char: char to be used to show identation level

    Returns
      A string containing the project structure.
    """
    startpath = self.path
    output = []
    for (root, dirs, files) in os.walk(startpath):
      level = root.replace(startpath, '').count(os.sep)
      indent = ident_char * tab_width * (level)
      if level == 0:
        output.append('{}{}/'.format(indent, os.path.basename(root)))
      else:
        output.append('|{}{}/'.format(indent, os.path.basename(root)))
      subindent = ident_char * tab_width * (level + 1)
      [output.append('|{}{}'.format(subindent, f)) for f in files]
    return '\n'.join(output)