def print_task_help(self, task, name):
        """
        Prints the help for the passed task with the passed name.
        :param task: the task function object
        :param name: the name of the module.
        :return: None
        """
        TerminalColor.set('GREEN')
        print(get_signature(name, task))

        # TODO: print the location does not work properly and sometimes returns None
        # print('    => defined in: {}'.format(inspect.getsourcefile(task)))
        help_msg = inspect.getdoc(task) or ''
        TerminalColor.reset()
        print('   ' + help_msg.replace('\n', '\n   '))
        TerminalColor.reset()
        print()