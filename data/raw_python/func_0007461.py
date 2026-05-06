def _delete_tasks(config, task_id, limit=100, offset=0):
    """Delete tasks from a project."""
    try:
        project = find_project_by_short_name(config.project['short_name'],
                                             config.pbclient,
                                             config.all)
        if task_id:
            response = config.pbclient.delete_task(task_id)
            check_api_error(response)
            return "Task.id = %s and its associated task_runs have been deleted" % task_id
        else:
            limit = limit
            offset = offset
            tasks = config.pbclient.get_tasks(project.id, limit, offset)
            while len(tasks) > 0:
                for t in tasks:
                    response = config.pbclient.delete_task(t.id)
                    check_api_error(response)
                offset += limit
                tasks = config.pbclient.get_tasks(project.id, limit, offset)
            return "All tasks and task_runs have been deleted"
    except exceptions.ConnectionError:
        return ("Connection Error! The server %s is not responding" % config.server)
    except (ProjectNotFound, TaskNotFound):
        raise