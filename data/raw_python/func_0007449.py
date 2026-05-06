def update_project(config, task_presenter, results,
                   long_description, tutorial, watch): # pragma: no cover
    """Update project templates and information."""
    if watch:
        res = _update_project_watch(config, task_presenter, results,
                                    long_description, tutorial)
    else:
        res = _update_project(config, task_presenter, results,
                              long_description, tutorial)
        click.echo(res)