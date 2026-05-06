def _pass_uid(self):
        """
        Purpose: Pass current Pipeline's uid to all Stages.

        :argument: List of Stage objects (optional)
        """

        for stage in self._stages:
            stage.parent_pipeline['uid'] = self._uid
            stage.parent_pipeline['name'] = self._name
            stage._pass_uid()