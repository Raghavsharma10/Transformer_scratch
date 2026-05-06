def main():
    # Create the pycons3rt directories
    try:
        initialize_pycons3rt_dirs()
    except OSError as ex:
        traceback.print_exc()
        return 1



    # Replace log directory paths
    log_dir_path = get_pycons3rt_log_dir() + os.path.sep
    conf_contents = default_logging_conf_file_contents.replace(replace_str, log_dir_path)

    # Create the logging config file
    logging_config_file_dest = os.path.join(get_pycons3rt_conf_dir(), 'pycons3rt-logging.conf')
    with open(logging_config_file_dest, 'w') as f:
        f.write(conf_contents)
    """
    
    for line in fileinput.input(logging_config_file_dest, inplace=True):
        if re.search(replace_str, line):
            new_line = re.sub(replace_str, log_dir_path, line, count=0)
            sys.stdout.write(new_line)
        else:
            sys.stdout.write(line)
    """
    return 0