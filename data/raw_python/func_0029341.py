def _get_handled_methods(self, actions_map):
        """ Get names of HTTP methods that can be used at requested URI.

        Arguments:
            :actions_map: Map of actions. Must have the same structure as
                self._item_actions and self._collection_actions
        """
        methods = ('OPTIONS',)

        defined_actions = []
        for action_name in actions_map.keys():
            view_method = getattr(self, action_name, None)
            method_exists = view_method is not None
            method_defined = view_method != self.not_allowed_action
            if method_exists and method_defined:
                defined_actions.append(action_name)

        for action in defined_actions:
            methods += actions_map[action]

        return methods