def migration_exists(self, app, fixture_path):
        """
        Return true if it looks like a migration already exists.
        """
        base_name = os.path.basename(fixture_path)
        # Loop through all migrations
        for migration_path in glob.glob(os.path.join(app.path, 'migrations', '*.py')):
            if base_name in open(migration_path).read():
                return True
        return False