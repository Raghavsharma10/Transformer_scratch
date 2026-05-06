def bootstrap(**kwargs):
    """ Bootstrap an EC2 instance that has been booted into an AMI from http://www.daemonology.net/freebsd-on-ec2/
    Note: deprecated, current AMI images are basically pre-bootstrapped, they just need to be configured.
    """
    # the user for the image is `ec2-user`, there is no sudo, but we can su to root w/o password
    original_host = env.host_string
    env.host_string = 'ec2-user@%s' % env.instance.uid
    bootstrap_files = env.instance.config.get('bootstrap-files', 'bootstrap-files')
    put('%s/authorized_keys' % bootstrap_files, '/tmp/authorized_keys')
    put(join(bsdploy_path, 'enable_root_login_on_daemonology.sh'), '/tmp/', mode='0775')
    run("""su root -c '/tmp/enable_root_login_on_daemonology.sh'""")
    # revert back to root
    env.host_string = original_host
    # give sshd a chance to restart
    sleep(2)
    run('rm /tmp/enable_root_login_on_daemonology.sh')

    # allow overwrites from the commandline
    env.instance.config.update(kwargs)

    bu = BootstrapUtils()
    bu.ssh_keys = None
    bu.upload_authorized_keys = False
    bu.bootstrap_files_yaml = 'daemonology-files.yml'
    bu.print_bootstrap_files()

    bu.create_bootstrap_directories()
    bu.upload_bootstrap_files({})
    # we need to install python here, because there is no way to install it via
    # ansible playbooks
    bu.install_pkg('/', chroot=False, packages=['python27'])