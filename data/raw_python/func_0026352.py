def list_matching_entries(program, arguments):
    """List the entries matching the given keywords/patterns."""
    output("\n".join(entry.name for entry in program.smart_search(*arguments)))