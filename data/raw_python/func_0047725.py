def start(name, config_file=None):
    '''
    starts a container in daemon mode
    '''
    if not exists(name):
        raise ContainerNotExists("The container (%s) does not exist!" % name)
    if name in running():
        raise ContainerAlreadyRunning('The container %s is already started!' % name)
    cmd = ['lxc-start', '-n', name, '-d']
    if config_file:
        cmd += ['-f', config_file]
    subprocess.check_call(cmd)