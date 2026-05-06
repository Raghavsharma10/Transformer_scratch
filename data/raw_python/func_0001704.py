def tasks(lancet, project_id):
    """List Harvest tasks for the given project ID."""
    for task in lancet.timer.tasks(project_id):
        click.echo('{:>9d} {} {}'.format(
            task['id'], click.style('‣', fg='yellow'), task['name']))