def run(interface, config, logfile, ros_args):
    """
    Start a pyros node.
    :param interface: the interface implementation (ROS, Mock, ZMP, etc.)
    :param config: the config file path, absolute, or relative to working directory
    :param logfile: the logfile path, absolute, or relative to working directory
    :param ros_args: the ros arguments (useful to absorb additional args when launched with roslaunch)
    """
    logging.info(
        'pyros started with : interface {interface} config {config} logfile {logfile} ros_args {ros_args}'.format(
            interface=interface, config=config, logfile=logfile, ros_args=ros_args))

    if interface == 'ros':
        node_proc = pyros_rosinterface_launch(node_name='pyros_rosinterface', pyros_config=config, ros_argv=ros_args)
    else:
        node_proc = None  # NOT IMPLEMENTED

    # node_proc.daemon = True  # we do NOT want a daemon(would stop when this main process exits...)
    client_conn = node_proc.start()