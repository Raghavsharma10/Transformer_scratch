def install_gem(gem):
    """ install a particular gem """
    with settings(hide('warnings', 'running', 'stdout', 'stderr'),
                  warn_only=False, capture=True):
        # convert 0 into True, any errors will always raise an exception
        return not bool(
            run("gem install %s --no-rdoc --no-ri" % gem).return_code)