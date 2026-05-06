def get_action_handler(self, controller_name, action_name):
        """
        Return action of controller as callable.

        If requested controller isn't found - return 'not_found' action
        of requested controller or Index controller.
        """
        try_actions = [
            controller_name + '/' + action_name,
            controller_name + '/not_found',
            # call Index controller to catch all unhandled pages
            'index/not_found'
        ]
        # search first appropriate action handler
        for path in try_actions:
            if path in self._controllers:
                return self._controllers[path]
        return None