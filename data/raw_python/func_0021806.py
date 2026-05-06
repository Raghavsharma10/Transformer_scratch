def purge_old_files(date_time, directory_path, custom_prefix="backup"):
    """ Takes a datetime object and a directory path, runs through files in the
        directory and deletes those tagged with a date from before the provided
        datetime.
        If your backups have a custom_prefix that is not the default ("backup"),
        provide it with the "custom_prefix" kwarg. """
    for file_name in listdir(directory_path):
        try:
            file_date_time = get_backup_file_time_tag(file_name, custom_prefix=custom_prefix)
        except ValueError as e:
            if "does not match format" in e.message:
                print("WARNING. file(s) in %s do not match naming convention."
                      % (directory_path))
                continue
            raise e
        if file_date_time < date_time:
            remove(directory_path + file_name)