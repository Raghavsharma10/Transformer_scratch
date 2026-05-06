def unfreeze(name):
    '''
    unfreezes the container
    '''
    if not exists(name):
        raise ContainerNotExists("The container (%s) does not exist!" % name)
    cmd = ['lxc-unfreeze', '-n', name]
    subprocess.check_call(cmd)