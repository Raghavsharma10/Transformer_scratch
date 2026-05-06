def install_container(container_name):
    """
    Installs the container specified by container_name
    :param container_name: string, name of the container
    """

    container_dir = os.path.join(os.environ['APE_ROOT_DIR'], container_name)
    if os.path.exists(container_dir):
        os.environ['CONTAINER_DIR'] = container_dir
    else:
        raise ContainerNotFound('ERROR: container directory not found: %s' % container_dir)

    install_script = os.path.join(container_dir, 'install.py')
    if os.path.exists(install_script):
        print('... running install.py for %s' % container_name)
        subprocess.check_call(['python', install_script])
    else:
        raise ContainerError('ERROR: this container does not provide an install.py!')