def superimpose(self, module):
        """
        superimpose a task module on registered tasks'''
        :param module: ape tasks module that is superimposed on available ape tasks
        :return: None
        """
        featuremonkey.compose(module, self._tasks)
        self._tasks.FEATURE_SELECTION.append(module.__name__)