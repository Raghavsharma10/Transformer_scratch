def import_data(self):
        """Run all of the import tasks.

        This is executed by the 'scripts.main.py' when the module is run as an
        executable. This can also be run as a method, in which case default
        arguments are loaded, but can be overriden using `**kwargs`.
        """

        tasks_list = self.load_task_list()
        warnings.filterwarnings(
            'ignore', r'Warning: converting a masked element to nan.')
        # FIX
        warnings.filterwarnings('ignore', category=DeprecationWarning)

        # Delete all old (previously constructed) output files
        if self.args.delete_old:
            self.log.warning("Deleting all old entry files.")
            self.delete_old_entry_files()

        # In update mode, load all entry stubs.
        if self.args.load_stubs or self.args.update:
            self.load_stubs()

        if self.args.travis:
            self.log.warning("Running in `travis` mode.")

        prev_priority = 0
        prev_task_name = ''
        # for task, task_obj in tasks_list.items():
        for task_name, task_obj in tasks_list.items():
            if not task_obj.active:
                continue
            self.log.warning("Task: '{}'".format(task_name))

            nice_name = task_obj.nice_name
            mod_name = task_obj.module
            func_name = task_obj.function
            priority = task_obj.priority

            # Make sure things are running in the correct order
            if priority < prev_priority and priority > 0:
                raise RuntimeError("Priority for '{}': '{}', less than prev,"
                                   "'{}': '{}'.\n{}"
                                   .format(task_name, priority, prev_task_name,
                                           prev_priority, task_obj))

            self.log.debug("\t{}, {}, {}, {}".format(nice_name, priority,
                                                     mod_name, func_name))
            mod = importlib.import_module('.' + mod_name, package='astrocats')
            self.current_task = task_obj
            getattr(mod, func_name)(self)

            num_events, num_stubs = self.count()
            self.log.warning("Task finished.  Events: {},  Stubs: {}".format(
                num_events, num_stubs))
            self.journal_entries()
            num_events, num_stubs = self.count()
            self.log.warning("Journal finished.  Events: {}, Stubs: {}".format(
                num_events, num_stubs))

            prev_priority = priority
            prev_task_name = task_name

        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss
        self.log.warning('Memory used (MBs): '
                         '{:,}'.format(memory / 1024. / 1024.))
        return