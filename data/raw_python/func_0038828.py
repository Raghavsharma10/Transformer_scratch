def command_dependants(options):
    """Command launched by CLI."""
    dependants = sorted(
        get_dependants(options.package.project_name),
        key=lambda n: n.lower()
    )

    if dependants:
        print(*dependants, sep='\n')