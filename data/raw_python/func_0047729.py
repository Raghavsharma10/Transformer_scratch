def destroy(name):
    '''
    removes a container [stops a container if it's running and]
    raises ContainerNotExists exception if the specified name is not created
    '''
    if not exists(name):
        raise ContainerNotExists("The container (%s) does not exist!" % name)
    cmd = ['lxc-destroy', '-f', '-n', name]
    subprocess.check_call(cmd)