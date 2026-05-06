def update():
    """Update to the latest pages."""
    repo_directory = get_config()['repo_directory']
    os.chdir(repo_directory)
    click.echo("Check for updates...")

    local = subprocess.check_output('git rev-parse master'.split()).strip()
    remote = subprocess.check_output(
        'git ls-remote https://github.com/tldr-pages/tldr/ HEAD'.split()
    ).split()[0]
    if local != remote:
        click.echo("Updating...")
        subprocess.check_call('git checkout master'.split())
        subprocess.check_call('git pull --rebase'.split())
        build_index()
        click.echo("Update to the latest and rebuild the index.")
    else:
        click.echo("No need for updates.")