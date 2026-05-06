def freeze(name):
    '''
    freezes the container
    '''
    if not exists(name):
        raise ContainerNotExists("The container (%s) does not exist!" % name)
    cmd = ['lxc-freeze', '-n', name]
    subprocess.check_call(cmd)