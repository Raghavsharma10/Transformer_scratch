def install_vagrant_plugin(plugin, use_sudo=False):
    """ install vagrant plugin """

    cmd = 'vagrant plugin install %s' % plugin

    with settings(hide('running', 'stdout')):
        if use_sudo:
            if plugin not in sudo('vagrant plugin list'):
                sudo(cmd)
        else:
            if plugin not in run('vagrant plugin list'):
                run(cmd)