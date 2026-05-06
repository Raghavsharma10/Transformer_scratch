def solvers(config_file, profile, solver_def, list_solvers):
    """Get solver details.

    Unless solver name/id specified, fetch and display details for
    all online solvers available on the configured endpoint.
    """

    with Client.from_config(
            config_file=config_file, profile=profile, solver=solver_def) as client:

        try:
            solvers = client.get_solvers(**client.default_solver)
        except SolverNotFoundError:
            click.echo("Solver(s) {} not found.".format(solver_def))
            return 1

        if list_solvers:
            for solver in solvers:
                click.echo(solver.id)
            return

        # ~YAML output
        for solver in solvers:
            click.echo("Solver: {}".format(solver.id))
            click.echo("  Parameters:")
            for name, val in sorted(solver.parameters.items()):
                click.echo("    {}: {}".format(name, strtrunc(val) if val else '?'))
            solver.properties.pop('parameters', None)
            click.echo("  Properties:")
            for name, val in sorted(solver.properties.items()):
                click.echo("    {}: {}".format(name, strtrunc(val)))
            click.echo("  Derived properties:")
            for name in sorted(solver.derived_properties):
                click.echo("    {}: {}".format(name, strtrunc(getattr(solver, name))))
            click.echo()