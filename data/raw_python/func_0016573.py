def sample(config_file, profile, solver_def, biases, couplings, random_problem,
           num_reads, verbose):
    """Submit Ising-formulated problem and return samples."""

    # TODO: de-dup wrt ping

    def echo(s, maxlen=100):
        click.echo(s if verbose else strtrunc(s, maxlen))

    try:
        client = Client.from_config(
            config_file=config_file, profile=profile, solver=solver_def)
    except Exception as e:
        click.echo("Invalid configuration: {}".format(e))
        return 1
    if config_file:
        echo("Using configuration file: {}".format(config_file))
    if profile:
        echo("Using profile: {}".format(profile))
    echo("Using endpoint: {}".format(client.endpoint))

    try:
        solver = client.get_solver()
    except SolverAuthenticationError:
        click.echo("Authentication error. Check credentials in your configuration file.")
        return 1
    except (InvalidAPIResponseError, UnsupportedSolverError):
        click.echo("Invalid or unexpected API response.")
        return 2
    except SolverNotFoundError:
        click.echo("Solver with the specified features does not exist.")
        return 3

    echo("Using solver: {}".format(solver.id))

    if random_problem:
        linear, quadratic = generate_random_ising_problem(solver)
    else:
        try:
            linear = ast.literal_eval(biases) if biases else []
        except Exception as e:
            click.echo("Invalid biases: {}".format(e))
        try:
            quadratic = ast.literal_eval(couplings) if couplings else {}
        except Exception as e:
            click.echo("Invalid couplings: {}".format(e))

    echo("Using qubit biases: {!r}".format(linear))
    echo("Using qubit couplings: {!r}".format(quadratic))
    echo("Number of samples: {}".format(num_reads))

    try:
        result = solver.sample_ising(linear, quadratic, num_reads=num_reads).result()
    except Exception as e:
        click.echo(e)
        return 4

    if verbose:
        click.echo("Result: {!r}".format(result))

    echo("Samples: {!r}".format(result['samples']))
    echo("Occurrences: {!r}".format(result['occurrences']))
    echo("Energies: {!r}".format(result['energies']))