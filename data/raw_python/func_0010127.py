def prompt_filetype(args):
    """Prompt user for filetype if none specified."""
    valid_types = ('print', 'text', 'csv', 'pdf', 'html')
    if not any(args[x] for x in valid_types):
        try:
            filetype = input('Print or save output as ({0}): '
                             .format(', '.join(valid_types))).lower()
            while filetype not in valid_types:
                filetype = input('Invalid entry. Choose from ({0}): '
                                 .format(', '.join(valid_types))).lower()
        except (KeyboardInterrupt, EOFError):
            return
        args[filetype] = True