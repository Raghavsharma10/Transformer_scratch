def render_from_repo(repo_path, to_path, template_params, settings_dir):
    """
    rendering all files into the target directory
    """
    TEMPLATE_PROJECT_FOLDER_PLACEHOLDER_NAME = 'deployer_project'

    repo_path = repo_path.rstrip('/')
    to_path = to_path.rstrip('/')
    files_to_render = get_template_filelist(repo_path, ignore_folders=[TEMPLATE_PROJECT_FOLDER_PLACEHOLDER_NAME])


    # rendering generic deploy files
    for single_file_path in files_to_render:
        source_file_path = single_file_path
        dest_file_path = source_file_path.replace(repo_path, to_path)

        render_from_single_file(source_file_path, dest_file_path, template_params)

    settings_template_dir = os.path.join(repo_path, TEMPLATE_PROJECT_FOLDER_PLACEHOLDER_NAME)
    settings_files = get_template_filelist(settings_template_dir)

    # rendering settings file
    for single_file_path in settings_files:
        source = single_file_path
        dest = single_file_path.replace(settings_template_dir, settings_dir)
        render_from_single_file(source, dest, template_params)