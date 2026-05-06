def create_virtualenv(venv_dir, use_venv_module=True):
        """
        creates a new virtualenv in venv_dir

        By default, the built-in venv module is used.
        On older versions of python, you may set use_venv_module to False to use virtualenv
        """

        if not use_venv_module:
            try:
                check_call(['virtualenv', venv_dir, '--no-site-packages'])
            except OSError:
                raise Exception('You probably dont have virtualenv installed: sudo apt-get install python-virtualenv')
        else:
            check_call([sys.executable or 'python', '-m', 'venv', venv_dir])