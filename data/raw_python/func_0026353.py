def show_matching_entry(program, arguments, use_clipboard=True, quiet=False, filters=()):
    """Show the matching entry on the terminal (and copy the password to the clipboard)."""
    entry = program.select_entry(*arguments)
    if not quiet:
        formatted_entry = entry.format_text(include_password=not use_clipboard, filters=filters)
        if formatted_entry and not formatted_entry.isspace():
            output(formatted_entry)
    if use_clipboard:
        entry.copy_password()