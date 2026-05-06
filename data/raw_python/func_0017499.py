def manager(self):
        """Integrate a Flask-Script."""
        from flask_script import Manager, Command

        manager = Manager(usage="Migrate database.")
        manager.add_command('create', Command(self.cmd_create))
        manager.add_command('migrate', Command(self.cmd_migrate))
        manager.add_command('rollback', Command(self.cmd_rollback))
        manager.add_command('list', Command(self.cmd_list))
        manager.add_command('merge', Command(self.cmd_merge))

        return manager