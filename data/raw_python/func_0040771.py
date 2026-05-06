def update(env):
    """
    Update an existing cipr project to the latest intalled version.
    """
    files = [path.join(env.project_directory, 'cipr.lua')]
    for filename in files:
        if path.exists(filename):
            os.remove(filename)
    app.command.run(['init', env.project_directory])