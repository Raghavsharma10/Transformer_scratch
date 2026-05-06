def _add_helpingmaterials(config, helping_file, helping_type):
    """Add helping materials to a project."""
    try:
        project = find_project_by_short_name(config.project['short_name'],
                                             config.pbclient,
                                             config.all)
        data = _load_data(helping_file, helping_type)
        if len(data) == 0:
            return ("Unknown format for the tasks file. Use json, csv, po or "
                    "properties.")
        # Show progress bar
        with click.progressbar(data, label="Adding Helping Materials") as pgbar:
            for d in pgbar:
                helping_info, file_path = create_helping_material_info(d)
                if file_path:
                    # Create first the media object
                    hm = config.pbclient.create_helpingmaterial(project_id=project.id,
                                                                info=helping_info,
                                                                file_path=file_path)
                    check_api_error(hm)

                    z = hm.info.copy()
                    z.update(helping_info)
                    hm.info = z
                    response = config.pbclient.update_helping_material(hm)
                    check_api_error(response)
                else:
                    response = config.pbclient.create_helpingmaterial(project_id=project.id,
                                                                      info=helping_info)
                check_api_error(response)
                # Check if for the data we have to auto-throttle task creation
                sleep, msg = enable_auto_throttling(config, data,
                                                    endpoint='/api/helpinmaterial')
                # If true, warn user
                if sleep:  # pragma: no cover
                    click.secho(msg, fg='yellow')
                # If auto-throttling enabled, sleep for sleep seconds
                if sleep:  # pragma: no cover
                    time.sleep(sleep)
            return ("%s helping materials added to project: %s" % (len(data),
                    config.project['short_name']))
    except exceptions.ConnectionError:
        return ("Connection Error! The server %s is not responding" % config.server)
    except (ProjectNotFound, TaskNotFound):
        raise