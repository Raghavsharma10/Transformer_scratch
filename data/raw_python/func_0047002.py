def run(
    categories, param_file, project_dir, plugin, target,
    status_update_interval
):
    """ Generate code for this project and run it """
    return _run(
        categories, param_file, project_dir, plugin, target,
        status_update_interval
    )