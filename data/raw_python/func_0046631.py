def setup():
    "Sets up the initial environment."
    parent, project = os.path.split(env.path)

    if not exists(parent):
        run('mkdir -p {0}'.format(parent))
        run('virtualenv {0}'.format(parent))

    with cd(parent):
        if not exists(project):
            run('git clone {repo_url} {project}'.format(project=project, **env))