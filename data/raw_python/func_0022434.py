def register_press_watcher(self, watcher_name, press_keys, *condition_list):
        """
        The watcher perform *press_keys* action sequentially when conditions match.
        """
        def unicode_to_list(a_unicode):
            a_list = list()
            comma_count = a_unicode.count(',')
            for count in range(comma_count + 1):
                comma_position = a_unicode.find(',')
                if comma_position == -1:
                    a_list.append(str(a_unicode))
                else:
                    a_list.append(a_unicode[0:comma_position])
                    a_unicode = a_unicode[comma_position + 1:]
            return a_list

        watcher = self.device.watcher(watcher_name)
        for condition in condition_list:
            watcher.when(**self.__unicode_to_dict(condition))
        watcher.press(*unicode_to_list(press_keys))
        self.device.watchers.run()