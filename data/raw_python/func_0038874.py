def command_locate(options):
    """Command launched by CLI."""
    matches = find_owners(options.file.name)

    if matches:
        print(*matches, sep='\n')