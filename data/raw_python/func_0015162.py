def is_local_subsection(command_dict):
    """Returns True if command dict is "local subsection", meaning
    that it is "if", "else" or "for" (not a real call, but calls
    run_section recursively."""
    for local_com in ['if ', 'for ', 'else ']:
        if list(command_dict.keys())[0].startswith(local_com):
            return True
    return False