def create_migration(self, app, fixture_path):
        """
        Create a data migration for app that uses fixture_path.
        """
        self.monkey_patch_migration_template(app, fixture_path)

        out = StringIO()
        management.call_command('makemigrations', app.label, empty=True, stdout=out)

        self.restore_migration_template()

        self.stdout.write(out.getvalue())