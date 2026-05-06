def get_backup_file_time_tag(file_name, custom_prefix="backup"):
    """ Returns a datetime object computed from a file name string, with
        formatting based on DATETIME_FORMAT."""
    name_string = file_name[len(custom_prefix):]
    time_tag = name_string.split(".", 1)[0]
    return datetime.strptime(time_tag, DATETIME_FORMAT)