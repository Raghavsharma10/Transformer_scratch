def get_possible_actions(self):
        """Return a sequence containing the possible actions that can be
        executed within the environment.

        Usage:
            possible_actions = scenario.get_possible_actions()

        Arguments: None
        Return:
            A sequence containing the possible actions which can be
            executed within this scenario.
        """
        possible_actions = self.wrapped.get_possible_actions()

        if len(possible_actions) <= 20:
            # Try to ensure that the possible actions are unique. Also, put
            # them into a list so we can iterate over them safely before
            # returning them; this avoids accidentally exhausting an
            # iterator, if the wrapped class happens to return one.
            try:
                possible_actions = list(set(possible_actions))
            except TypeError:
                possible_actions = list(possible_actions)

            try:
                possible_actions.sort()
            except TypeError:
                pass

            self.logger.info('Possible actions:')
            for action in possible_actions:
                self.logger.info('    %s', action)
        else:
            self.logger.info("%d possible actions.", len(possible_actions))

        return possible_actions