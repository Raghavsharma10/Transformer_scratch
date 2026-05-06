def get_commands_from_dir(directory, zip_backup=True, remove_dir=True):
    """Traverse a directory and read contained SQL files."""
    # Get SQL commands file paths
    failed_scripts = sorted([os.path.join(directory, fn) for fn in os.listdir(directory) if fn.endswith('.sql')])

    # Read each failed SQL file and append contents to a list
    print('\tReading SQL scripts from files')
    commands = []
    for sql_file in failed_scripts:
        with open(sql_file, 'r') as txt:
            sql_command = txt.read()
        commands.append(sql_command)

    # Remove most recent failures folder after reading
    if zip_backup:
        ZipBackup(directory).backup()
    if remove_dir:
        shutil.rmtree(directory)
    return commands