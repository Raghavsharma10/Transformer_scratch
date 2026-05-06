def create(name, config_file=None, template=None, backing_store=None, template_options=None):
    '''
    Create a new container
    raises ContainerAlreadyExists exception if the container name is reserved already.
    
    :param template_options: Options passed to the specified template
    :type template_options: list or None
    
    '''
    if exists(name):
        raise ContainerAlreadyExists("The Container %s is already created!" % name)
    cmd = 'lxc-create -n %s' % name

    if config_file:
        cmd += ' -f %s' % config_file
    if template:
        cmd += ' -t %s' % template
    if backing_store:
        cmd += ' -B %s' % backing_store
    if template_options:
        cmd += '-- %s' % template_options

    if subprocess.check_call('%s >> /dev/null' % cmd, shell=True) == 0:
        if not exists(name):
            _logger.critical("The Container %s doesn't seem to be created! (options: %s)", name, cmd[3:])
            raise ContainerNotExists("The container (%s) does not exist!" % name)

        _logger.info("Container %s has been created with options %s", name, cmd[3:])
        return 0
    else:
        return 1