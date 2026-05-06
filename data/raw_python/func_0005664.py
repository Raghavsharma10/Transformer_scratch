def add_hooks():
    # type: () -> None
    """ Add git hooks for commit and push to run linting and tests. """

    # Detect virtualenv the hooks should use

    # Detect virtualenv
    virtual_env = conf.getenv('VIRTUAL_ENV')
    if virtual_env is None:
        log.err("You are not inside a virtualenv")
        confirm_msg = (
            "Are you sure you want to use global python installation "
            "to run your git hooks? [y/N] "
        )
        click.prompt(confirm_msg, default=False)
        if not click.confirm(confirm_msg):
            log.info("Cancelling")
            return

        load_venv = ''
    else:
        load_venv = 'source "{}/bin/activate"'.format(virtual_env)

    commit_hook = conf.proj_path('.git/hooks/pre-commit')
    push_hook = conf.proj_path('.git/hooks/pre-push')

    # Write pre-commit hook
    log.info("Adding pre-commit hook <33>{}", commit_hook)
    fs.write_file(commit_hook, util.remove_indent('''
        #!/bin/bash
        PATH="/opt/local/libexec/gnubin:$PATH"
        
        {load_venv}
        
        peltak lint --commit
        
    '''.format(load_venv=load_venv)))

    # Write pre-push hook
    log.info("Adding pre-push hook: <33>{}", push_hook)
    fs.write_file(push_hook, util.remove_indent('''
        #!/bin/bash
        PATH="/opt/local/libexec/gnubin:$PATH"
        
        {load_venv}
        
        peltak test --allow-empty
        
    '''.format(load_venv=load_venv)))

    log.info("Making hooks executable")
    if not context.get('pretend', False):
        os.chmod(conf.proj_path('.git/hooks/pre-commit'), 0o755)
        os.chmod(conf.proj_path('.git/hooks/pre-push'), 0o755)