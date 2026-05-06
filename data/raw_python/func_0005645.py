def project_dev_requirements():
    """ List requirements for peltak commands configured for the project.

    This list is dynamic and depends on the commands you have configured in
    your project's pelconf.yaml. This will be the combined list of packages
    needed to be installed in order for all the configured commands to work.
    """
    from peltak.core import conf
    from peltak.core import shell

    for dep in sorted(conf.requirements):
        shell.cprint(dep)