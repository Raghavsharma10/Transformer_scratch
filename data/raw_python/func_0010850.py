def load_migrations(self):  # pragma: no cover
        """
        Loads all migrations in the order they would be applied to a clean database
        """
        executor = MigrationExecutor(connection=None)

        # create the forwards plan Django would follow on an empty database
        plan = executor.migration_plan(executor.loader.graph.leaf_nodes(), clean_start=True)

        if self.verbosity >= 2:
            for migration, _ in plan:
                self.stdout.write(" > %s" % migration)

        return [m[0] for m in plan]