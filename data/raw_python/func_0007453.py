def update_task_redundancy(config, task_id, redundancy):
    """Update task redudancy for a project."""
    if task_id is None:
        msg = ("Are you sure you want to update all the tasks redundancy?")
        if click.confirm(msg):
            res = _update_tasks_redundancy(config, task_id, redundancy)
            click.echo(res)

        else:
            click.echo("Aborting.")
    else:
        res = _update_tasks_redundancy(config, task_id, redundancy)
        click.echo(res)