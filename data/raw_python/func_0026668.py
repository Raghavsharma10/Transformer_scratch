def load_task_list(self):
        """Load the list of tasks in this catalog's 'input/tasks.json' file.

        A `Task` object is created for each entry, with the parameters filled
        in. These are placed in an OrderedDict, sorted by the `priority`
        parameter, with positive values and then negative values,
            e.g. [0, 2, 10, -10, -1].
        """
        # In update mode, do not delete old files
        if self.args.update:
            self.log.info("Disabling `pre-delete` for 'update' mode.")
            self.args.delete_old = False

        # Dont allow both a 'min' and 'max' task priority
        # FIX: this is probably unnecessary... having both could be useful
        if ((self.args.min_task_priority is not None and
             self.args.max_task_priority is not None)):
            raise ValueError("Can only use *either* 'min' *or* 'max' priority")

        # Load tasks data from input json file
        tasks, task_names = self._load_task_list_from_file()

        # Make sure 'active' modification lists are all valid
        args_lists = [
            self.args.args_task_list, self.args.yes_task_list,
            self.args.no_task_list
        ]
        args_names = ['--tasks', '--yes', '--no']
        for arglist, lname in zip(args_lists, args_names):
            if arglist is not None:
                for tname in arglist:
                    if tname not in task_names:
                        raise ValueError(
                            "Value '{}' in '{}' list does not match"
                            " any tasks".format(tname, lname))

        # Process min/max priority specification ('None' if none given)
        min_priority = _get_task_priority(tasks, self.args.min_task_priority)
        max_priority = _get_task_priority(tasks, self.args.max_task_priority)
        task_groups = self.args.task_groups
        if task_groups is not None:
            if not isinstance(task_groups, list):
                task_groups = [task_groups]

        # Iterate over all tasks to determine which should be (in)active
        # --------------------------------------------------------------
        for key in tasks:
            # If in update mode, only run update tasks
            if self.args.update:
                if not tasks[key].update:
                    tasks[key].active = False

            # If specific list of tasks is given, make only those active
            if self.args.args_task_list is not None:
                if key in self.args.args_task_list:
                    tasks[key].active = True
                else:
                    tasks[key].active = False

            # Only run tasks above minimum priority
            # (doesn't modify negtive priority tasks)
            if min_priority is not None and tasks[key].priority >= 0:
                tasks[key].active = False
                if tasks[key].priority >= min_priority:
                    tasks[key].active = True

            # Only run tasks below maximum priority
            # (doesnt modify negative priority tasks)
            if max_priority is not None and tasks[key].priority >= 0:
                tasks[key].active = False
                if tasks[key].priority <= max_priority:
                    tasks[key].active = True

            # Set 'yes' tasks to *active*
            if self.args.yes_task_list is not None:
                if key in self.args.yes_task_list:
                    tasks[key].active = True
            # Set 'no' tasks to *inactive*
            if self.args.no_task_list is not None:
                if key in self.args.no_task_list:
                    tasks[key].active = False
            # Set tasks in target 'groups' to *active*
            if task_groups is not None and tasks[key].groups is not None:
                # Go through each group defined in the command line
                for given_group in task_groups:
                    # If this task is a member of any of those groups
                    if given_group in tasks[key].groups:
                        tasks[key].active = True
                        break

        # Sort entries as positive values, then negative values
        #    [0, 1, 2, 2, 10, -100, -10, -1]
        # Tuples are sorted by first element (here: '0' if positive), then
        # second (here normal order)
        tasks = OrderedDict(
            sorted(
                tasks.items(),
                key=lambda t: (t[1].priority < 0, t[1].priority, t[1].name)))

        # Find the first task that has "always_journal" set to True
        for key in tasks:
            if tasks[key].active and tasks[key].always_journal:
                self.min_journal_priority = tasks[key].priority
                break

        names_act = []
        names_inact = []
        for key, val in tasks.items():
            if val.active:
                names_act.append(key)
            else:
                names_inact.append(key)

        self.log.info("Active Tasks:\n\t" + ", ".join(nn for nn in names_act))
        self.log.debug("Inactive Tasks:\n\t" + ", ".join(nn for nn in
                                                         names_inact))
        return tasks