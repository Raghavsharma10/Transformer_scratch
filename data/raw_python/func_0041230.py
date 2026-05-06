def _run__group(self, action, replace):
        """
        Run a group of actions in sequence.

        >>> Action().run("several", actions={
        ...     "several": {
        ...         "type": "group",
        ...         "actions": ["hello","call","then"]
        ...     }, "hello": {
        ...         "type": "exec",
        ...         "cmd": "echo version=%{version}"
        ...     }, "call": {
        ...         "type": "hook",
        ...         "url": "http://reflex.cold.org"
        ...     }, "then": {
        ...         "type": "exec",
        ...         "cmd": "echo finished"
        ... }}, replace={
        ...     "version": "1712.10"
        ... })
        version=1712.10
        """

        for target in action.get('actions', []):
            Action().run(target, actions=self.actions, replace=replace)