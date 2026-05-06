def _add_tasks(config, tasks_file, tasks_type, priority, redundancy):
    """Add tasks to a project."""
    try:
        project = find_project_by_short_name(config.project['short_name'],
                                             config.pbclient,
                                             config.all)
        data = _load_data(tasks_file, tasks_type)
        if len(data) == 0:
            return ("Unknown format for the tasks file. Use json, csv, po or "
                    "properties.")
        # If true, warn user
        # if sleep:  # pragma: no cover
        #     click.secho(msg, fg='yellow')
        # Show progress bar
        with click.progressbar(data, label="Adding Tasks") as pgbar:
            for d in pgbar:
                task_info = create_task_info(d)
                response = config.pbclient.create_task(project_id=project.id,
                                                       info=task_info,
                                                       n_answers=redundancy,
                                                       priority_0=priority)

                # Check if for the data we have to auto-throttle task creation
                sleep, msg = enable_auto_throttling(config, data)
                check_api_error(response)
                # If auto-throttling enabled, sleep for sleep seconds
                if sleep:  # pragma: no cover
                    time.sleep(sleep)
            return ("%s tasks added to project: %s" % (len(data),
                    config.project['short_name']))
    except exceptions.ConnectionError:
        return ("Connection Error! The server %s is not responding" % config.server)
    except (ProjectNotFound, TaskNotFound):
        raise