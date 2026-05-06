def edit_matching_entry(program, arguments):
    """Edit the matching entry."""
    entry = program.select_entry(*arguments)
    entry.context.execute("pass", "edit", entry.name)