def _prepare_docstrings():
    """Assign docstrings to the corresponding attributes of class `Options`
     to make them available in the interactive mode of Python."""
    if config.USEAUTODOC:
        source = inspect.getsource(Options)
        docstrings = source.split('"""')[3::2]
        attributes = [line.strip().split()[0] for line in source.split('\n')
                      if '_Option(' in line]
        for attribute, docstring in zip(attributes, docstrings):
            Options.__dict__[attribute].__doc__ = docstring