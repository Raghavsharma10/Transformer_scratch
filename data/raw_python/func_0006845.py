def add_listener(self, evt_name, fn):
        """添加观察者函数。

        :params evt_name: 事件名称
        :params fn: 要注册的触发函数函数

        .. note::
           允许一个函数多次注册，多次注册意味着一次 :func:`fire_event` 多次调用。
        """
        self._listeners.setdefault(evt_name, [])
        listeners = self.__get_listeners(evt_name)
        listeners.append(fn)