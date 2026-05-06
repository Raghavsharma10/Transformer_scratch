def get_version_exec_mapping_from_path(path):
    """
    Find valid application version from given path object and return
    a mapping of version, executable.
    """
    version_executable = {}
    logger.debug('Getting exes from path: {}'.format(path))

    for sub_dir in path.iterdir():
        if not sub_dir.name.startswith(APPLICATION_NAME):
            continue

        release = sub_dir.name.split(APPLICATION_NAME)[-1]
        executable = Path(sub_dir, 'bin').glob('maya.exe').next()
        version_executable[release] = str(executable)
    logger.debug('Found exes for: {}'.format(version_executable.keys()))
    return version_executable