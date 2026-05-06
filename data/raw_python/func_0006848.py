def fire_event(self, evt_name, *args, **kwargs):
        """触发事件

        :params evt_name: 事件名称
        :params args: 给事件接受者的参数
        :params kwargs: 给事件接受者的参数
        """
        listeners = self.__get_listeners(evt_name)
        evt = self.generate_event(evt_name)
        for listener in listeners:
            listener(evt, *args, **kwargs)