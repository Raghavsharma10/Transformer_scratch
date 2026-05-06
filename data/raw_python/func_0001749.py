def setup(ctx, force):
    """Wizard to create the user-level configuration file."""
    if os.path.exists(USER_CONFIG) and not force:
        click.secho(
            'An existing configuration file was found at "{}".\n'
            .format(USER_CONFIG),
            fg='red', bold=True
        )
        click.secho(
            'Please remove it before in order to run the setup wizard or use\n'
            'the --force flag to overwrite it.'
        )
        ctx.exit(1)

    click.echo('Address of the issue tracker (your JIRA instance). \n'
               'Normally in the form https://<company>.atlassian.net.')
    tracker_url = click.prompt('URL')
    tracker_user = click.prompt('Username for {}'.format(tracker_url))
    click.echo()

    click.echo('Address of the time tracker (your Harvest instance). \n'
               'Normally in the form https://<company>.harvestapp.com.')
    timer_url = click.prompt('URL')
    timer_user = click.prompt('Username for {}'.format(timer_url))
    click.echo()

    config = configparser.ConfigParser()

    config.add_section('tracker')
    config.set('tracker', 'url', tracker_url)
    config.set('tracker', 'username', tracker_user)

    config.add_section('harvest')
    config.set('harvest', 'url', timer_url)
    config.set('harvest', 'username', timer_user)

    with open(USER_CONFIG, 'w') as fh:
        config.write(fh)

    click.secho('Configuration correctly written to "{}".'
                .format(USER_CONFIG), fg='green')