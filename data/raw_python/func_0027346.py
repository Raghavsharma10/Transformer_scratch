def monkey_patch_migration_template(self, app, fixture_path):
        """
        Monkey patch the django.db.migrations.writer.MIGRATION_TEMPLATE

        Monkey patching django.db.migrations.writer.MIGRATION_TEMPLATE means that we 
        don't have to do any complex regex or reflection.

        It's hacky... but works atm.
        """
        self._MIGRATION_TEMPLATE = writer.MIGRATION_TEMPLATE
        module_split = app.module.__name__.split('.')

        if len(module_split) == 1:
            module_import = "import %s\n" % module_split[0]
        else:
            module_import = "from %s import %s\n" % (
                '.'.join(module_split[:-1]),
                module_split[-1:][0],
            )

        writer.MIGRATION_TEMPLATE = writer.MIGRATION_TEMPLATE\
            .replace(
                '%(imports)s',
                "%(imports)s" + "\nfrom django_migration_fixture import fixture\n%s" % module_import
            )\
            .replace(
                '%(operations)s', 
                "        migrations.RunPython(**fixture(%s, ['%s'])),\n" % (
                    app.label,
                    os.path.basename(fixture_path)
                ) + "%(operations)s\n" 
            )