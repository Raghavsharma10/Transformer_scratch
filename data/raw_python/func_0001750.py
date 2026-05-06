def init(ctx, force):
    """Wizard to create a project-level configuration file."""
    if os.path.exists(PROJECT_CONFIG) and not force:
        click.secho(
            'An existing configuration file was found at "{}".\n'
            .format(PROJECT_CONFIG),
            fg='red', bold=True
        )
        click.secho(
            'Please remove it before in order to run the setup wizard or use\n'
            'the --force flag to overwrite it.'
        )
        ctx.exit(1)

    project_key = click.prompt('Project key on the issue tracker')
    base_branch = click.prompt('Integration branch', default='master')

    virtualenvs = ('.venv', '.env', 'venv', 'env')
    for p in virtualenvs:
        if os.path.exists(os.path.join(p, 'bin', 'activate')):
            venv = p
            break
    else:
        venv = ''
    venv_path = click.prompt('Path to virtual environment', default=venv)

    project_id = click.prompt('Project ID on Harvest', type=int)
    task_id = click.prompt('Task id on Harvest', type=int)

    config = configparser.ConfigParser()

    config.add_section('lancet')
    config.set('lancet', 'virtualenv', venv_path)

    config.add_section('tracker')
    config.set('tracker', 'default_project', project_key)

    config.add_section('harvest')
    config.set('harvest', 'project_id', str(project_id))
    config.set('harvest', 'task_id', str(task_id))

    config.add_section('repository')
    config.set('repository', 'base_branch', base_branch)

    with open(PROJECT_CONFIG, 'w') as fh:
        config.write(fh)

    click.secho('\nConfiguration correctly written to "{}".'
                .format(PROJECT_CONFIG), fg='green')