def _add_parser_arguments_import(self, subparsers):
        """Create parser for 'import' subcommand, and associated arguments.
        """
        import_pars = subparsers.add_parser(
            "import", help="Import data.")

        import_pars.add_argument(
            '--update', '-u', dest='update',
            default=False, action='store_true',
            help='Only update catalog using live sources.')
        import_pars.add_argument(
            '--load-stubs', dest='load_stubs',
            default=False, action='store_true',
            help='Load stubs before running.')
        import_pars.add_argument(
            '--archived', '-a', dest='archived',
            default=False, action='store_true',
            help='Always use task caches.')

        # Control which 'tasks' are executed
        # ----------------------------------
        import_pars.add_argument(
            '--tasks', dest='args_task_list', nargs='*', default=None,
            help='space delimited list of tasks to perform.')
        import_pars.add_argument(
            '--yes', dest='yes_task_list', nargs='+', default=None,
            help='space delimited list of tasks to turn on.')
        import_pars.add_argument(
            '--no', dest='no_task_list', nargs='+', default=None,
            help='space delimited list of tasks to turn off.')
        import_pars.add_argument(
            '--min-task-priority', dest='min_task_priority',
            default=None,
            help='minimum priority for a task to run')
        import_pars.add_argument(
            '--max-task-priority', dest='max_task_priority',
            default=None,
            help='maximum priority for a task to run')
        import_pars.add_argument(
            '--task-groups', dest='task_groups',
            default=None,
            help='predefined group(s) of tasks to run.')

        return import_pars