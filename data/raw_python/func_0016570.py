def _ping(config_file, profile, solver_def, request_timeout, polling_timeout, output):
    """Helper method for the ping command that uses `output()` for info output
    and raises `CLIError()` on handled errors.

    This function is invariant to output format and/or error signaling mechanism.
    """

    config = dict(config_file=config_file, profile=profile, solver=solver_def)
    if request_timeout is not None:
        config.update(request_timeout=request_timeout)
    if polling_timeout is not None:
        config.update(polling_timeout=polling_timeout)
    try:
        client = Client.from_config(**config)
    except Exception as e:
        raise CLIError("Invalid configuration: {}".format(e), code=1)
    if config_file:
        output("Using configuration file: {config_file}", config_file=config_file)
    if profile:
        output("Using profile: {profile}", profile=profile)
    output("Using endpoint: {endpoint}", endpoint=client.endpoint)

    t0 = timer()
    try:
        solver = client.get_solver()
    except SolverAuthenticationError:
        raise CLIError("Authentication error. Check credentials in your configuration file.", 2)
    except SolverNotFoundError:
        raise CLIError("Solver not available.", 6)
    except (InvalidAPIResponseError, UnsupportedSolverError):
        raise CLIError("Invalid or unexpected API response.", 3)
    except RequestTimeout:
        raise CLIError("API connection timed out.", 4)
    except requests.exceptions.SSLError as e:
        # we need to handle `ssl.SSLError` wrapped in several exceptions,
        # with differences between py2/3; greping the message is the easiest way
        if 'CERTIFICATE_VERIFY_FAILED' in str(e):
            raise CLIError(
                "Certificate verification failed. Please check that your API endpoint "
                "is correct. If you are connecting to a private or third-party D-Wave "
                "system that uses self-signed certificate(s), please see "
                "https://support.dwavesys.com/hc/en-us/community/posts/360018930954.", 5)
        raise CLIError("Unexpected SSL error while fetching solver: {!r}".format(e), 5)
    except Exception as e:
        raise CLIError("Unexpected error while fetching solver: {!r}".format(e), 5)

    t1 = timer()
    output("Using solver: {solver_id}", solver_id=solver.id)

    try:
        future = solver.sample_ising({0: 1}, {})
        timing = future.timing
    except RequestTimeout:
        raise CLIError("API connection timed out.", 8)
    except PollingTimeout:
        raise CLIError("Polling timeout exceeded.", 9)
    except Exception as e:
        raise CLIError("Sampling error: {!r}".format(e), 10)
    finally:
        output("Submitted problem ID: {problem_id}", problem_id=future.id)

    t2 = timer()

    output("\nWall clock time:")
    output(" * Solver definition fetch: {wallclock_solver_definition:.3f} ms", wallclock_solver_definition=(t1-t0)*1000.0)
    output(" * Problem submit and results fetch: {wallclock_sampling:.3f} ms", wallclock_sampling=(t2-t1)*1000.0)
    output(" * Total: {wallclock_total:.3f} ms", wallclock_total=(t2-t0)*1000.0)
    if timing.items():
        output("\nQPU timing:")
        for component, duration in timing.items():
            output(" * %(name)s = {%(name)s} us" % {"name": component}, **{component: duration})
    else:
        output("\nQPU timing data not available.")