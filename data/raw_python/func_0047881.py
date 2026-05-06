def extra_prepare_after_activation(self, configuration, args_dict):
        """
        Called after the configuration.converters are activated

        Here we create our ``task_maker`` helper that we pass into ``post_register``
        for our ``option_merge_addon_hook`` functions.

        We also create a ``task_finder`` for doing task finding related duties.
        """
        def task_maker(name, description=None, action=None, label="Project", **options):
            if not action:
                action = name
            self.task_overrides[name] = Task(action=action, description=description, options=options, label=label)
            return self.task_overrides[name]

        # Post register our addons
        extra_args = {"harpoon.crosshairs": {"task_maker": task_maker}}
        self.register.post_register(extra_args)

        # Make the task finder
        task_finder = TaskFinder(self)
        configuration["task_runner"] = task_finder.task_runner
        task_finder.find_tasks(self.task_overrides)