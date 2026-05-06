def delete_tasks(config, task_id):
    """Delete tasks from a project."""
    if task_id is None:
        msg = ("Are you sure you want to delete all the tasks and associated task runs?")
        if click.confirm(msg):
            res = _delete_tasks(config, task_id)
            click.echo(res)

        else:
            click.echo("Aborting.")
    else:
        res = _delete_tasks(config, task_id)
        click.echo(res)