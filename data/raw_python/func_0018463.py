def override_env_variables():
    """Override user environmental variables with custom one."""
    env_vars = ("LOGNAME", "USER", "LNAME", "USERNAME")
    old = [os.environ[v] if v in os.environ else None for v in env_vars]

    for v in env_vars:
        os.environ[v] = "test"
    yield

    for i, v in enumerate(env_vars):
        if old[i]:
            os.environ[v] = old[i]