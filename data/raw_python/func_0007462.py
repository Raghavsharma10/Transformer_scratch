def _update_tasks_redundancy(config, task_id, redundancy, limit=300, offset=0):
    """Update tasks redundancy from a project."""
    try:
        project = find_project_by_short_name(config.project['short_name'],
                                             config.pbclient,
                                             config.all)
        if task_id:
            response = config.pbclient.find_tasks(project.id, id=task_id)
            check_api_error(response)
            task = response[0]
            task.n_answers = redundancy
            response = config.pbclient.update_task(task)
            check_api_error(response)
            msg = "Task.id = %s redundancy has been updated to %s" % (task_id,
                                                                      redundancy)
            return msg
        else:
            limit = limit
            offset = offset
            tasks = config.pbclient.get_tasks(project.id, limit, offset)
            with click.progressbar(tasks, label="Updating Tasks") as pgbar:
                while len(tasks) > 0:
                    for t in pgbar:
                        t.n_answers = redundancy
                        response = config.pbclient.update_task(t)
                        check_api_error(response)
                        # Check if for the data we have to auto-throttle task update
                        sleep, msg = enable_auto_throttling(config, tasks)
                        # If auto-throttling enabled, sleep for sleep seconds
                        if sleep:  # pragma: no cover
                            time.sleep(sleep)
                    offset += limit
                    tasks = config.pbclient.get_tasks(project.id, limit, offset)
                return "All tasks redundancy have been updated"
    except exceptions.ConnectionError:
        return ("Connection Error! The server %s is not responding" % config.server)
    except (ProjectNotFound, TaskNotFound):
        raise