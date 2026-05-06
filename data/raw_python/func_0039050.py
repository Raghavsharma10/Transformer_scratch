def command_dependents(options):
    """Command launched by CLI."""
    dependents = dependencies(options.package, options.recursive, options.info)

    if dependents:
        print(*dependents, sep='\n')