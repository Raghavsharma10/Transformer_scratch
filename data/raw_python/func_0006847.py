def has_listener(self, evt_name, fn):
        """指定listener是否存在

        :params evt_name: 事件名称
        :params fn: 要注册的触发函数函数
        """
        listeners = self.__get_listeners(evt_name)
        return fn in listeners