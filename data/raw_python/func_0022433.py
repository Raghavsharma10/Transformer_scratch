def register_click_watcher(self, watcher_name, selectors, *condition_list):
        """
        The watcher click on the object which has the *selectors* when conditions match.
        """
        watcher = self.device.watcher(watcher_name)
        for condition in condition_list:
            watcher.when(**self.__unicode_to_dict(condition))
        watcher.click(**self.__unicode_to_dict(selectors))
        self.device.watchers.run()