def add_arguments(self, actions):
        """
        Sort the flags alphabetically
        """
        actions = sorted(
            actions, key=operator.attrgetter('option_strings'))
        super(SortedHelpFormatter, self).add_arguments(actions)