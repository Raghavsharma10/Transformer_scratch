def command_shell(self):
        '''
        Shell command::

            ./manage.py app:shell

        Executed with `self.shell_namespace` as local variables namespace.
        '''
        from code import interact
        interact('Namespace {!r}'.format(self.shell_namespace),
                 local=self.shell_namespace)