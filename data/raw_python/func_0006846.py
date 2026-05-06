def remove_listener(self, evt_name, fn, remove_all=False):
        """删除观察者函数。

        :params evt_name: 事件名称
        :params fn: 要注册的触发函数函数
        :params remove_all: 是否删除fn在evt_name中的所有注册\n
                            如果为 `True`，则删除所有\n
                            如果为 `False`，则按注册先后顺序删除第一个\n

        .. note::
           允许一个函数多次注册，多次注册意味着一次时间多次调用。
        """
        listeners = self.__get_listeners(evt_name)
        if not self.has_listener(evt_name, fn):
            raise ObservableError(
                "function %r does not exist in the %r event",
                fn, evt_name)
        if remove_all:
            listeners[:] = [i for i in listeners if i != fn]
        else:
            listeners.remove(fn)