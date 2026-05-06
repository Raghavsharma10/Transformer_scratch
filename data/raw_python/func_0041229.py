def run(self, name, replace=None, actions=None):
        """
        Do an action.


        If `replace` is provided as a dictionary, do a search/replace using
        %{} templates on content of action (unique to action type)
        """
        self.actions = actions # incase we use group

        action = actions.get(name)
        if not action:
            self.die("Action not found: {}", name)
        action['name'] = name
        action_type = action.get('type', "none")
        try:
            func = getattr(self, '_run__' + action_type)
        except AttributeError:
            self.die("Unsupported action type " + action_type)
        try:
            return func(action, replace)
        except Exception as err: # pylint: disable=broad-except
            if self._debug:
                self.debug(traceback.format_exc())
            self.die("Error running action name={} type={} error={}",
                     name, action_type, err)