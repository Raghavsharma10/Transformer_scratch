def CheckDirectory(self, path, extension='yaml'):
    """Validates definition files in a directory.

    Args:
      path (str): path of the definition file.
      extension (Optional[str]): extension of the filenames to read.

    Returns:
      bool: True if the directory contains valid definitions.
    """
    result = True

    if extension:
      glob_spec = os.path.join(path, '*.{0:s}'.format(extension))
    else:
      glob_spec = os.path.join(path, '*')

    for definition_file in sorted(glob.glob(glob_spec)):
      if not self.CheckFile(definition_file):
        result = False

    return result