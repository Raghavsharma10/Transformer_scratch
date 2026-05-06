def common_install_python_env(self):
        """
            Install python virtualenv
        """
        sudo('apt-get install python3 python3-pip -y')
        sudo('pip3 install virtualenv')

        run('virtualenv {0}'.format(self.python_env_dir))

        print(green(' * Installed Python3 virtual environment in the system.'))
        print(green(' * Done'))
        print()