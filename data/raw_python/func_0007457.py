def _update_project(config, task_presenter, results,
                    long_description, tutorial):
    """Update a project."""
    try:
        # Get project
        project = find_project_by_short_name(config.project['short_name'],
                                             config.pbclient,
                                             config.all)
        # Update attributes
        project.name = config.project['name']
        project.short_name = config.project['short_name']
        project.description = config.project['description']
        # Update long_description
        with open(long_description, 'r') as f:
            project.long_description = f.read()
        # Update task presenter
        with open(task_presenter, 'r') as f:
            project.info['task_presenter'] = f.read()
        _update_task_presenter_bundle_js(project)
        # Update results
        with open(results, 'r') as f:
            project.info['results'] = f.read()
        # Update tutorial
        with open(tutorial, 'r') as f:
            project.info['tutorial'] = f.read()
        response = config.pbclient.update_project(project)
        check_api_error(response)
        return ("Project %s updated!" % config.project['short_name'])
    except exceptions.ConnectionError:
        return ("Connection Error! The server %s is not responding" % config.server)
    except ProjectNotFound:
        return ("Project not found! The project: %s is missing." \
                " Use the flag --all=1 to search in all the server " \
                % config.project['short_name'])
    except TaskNotFound:
        raise