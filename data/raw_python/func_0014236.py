def get_action_by_dest(self, parser, dest):
        '''Retrieves the given parser action object by its dest= attribute'''
        for action in parser._actions:
            if action.dest == dest:
                return action
        return None