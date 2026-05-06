def init_tasks():
    """
    Performs basic setup before any of the tasks are run. All tasks needs to
    run this before continuing. It only fires once.
    """

    # Make sure exist are set
    if "exists" not in env:
        env.exists = exists

    if "run" not in env:
        env.run = run

    if "cd" not in env:
        env.cd = cd

    if "max_releases" not in env:
        env.max_releases = 5

    if "public_path" in env:
        public_path = env.public_path.rstrip("/")
        env.public_path = public_path

    run_hook("init_tasks")