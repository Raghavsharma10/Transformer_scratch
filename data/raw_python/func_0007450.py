def add_tasks(config, tasks_file, tasks_type, priority, redundancy):
    """Add tasks to a project."""
    res = _add_tasks(config, tasks_file, tasks_type, priority, redundancy)
    click.echo(res)