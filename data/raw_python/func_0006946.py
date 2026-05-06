def install_system_gem(gem):
    """ install a particular gem """
    with settings(hide('warnings', 'running', 'stdout', 'stderr'),
                  warn_only=False, capture=True):
        sudo("gem install %s --no-rdoc --no-ri" % gem)