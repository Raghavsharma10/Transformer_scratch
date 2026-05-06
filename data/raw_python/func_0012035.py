def Alt(cls, key):
        """    在指定元素上执行alt组合事件
        @note: key event ->  alt + key
        @param key: 如'X'
        """
        element = cls._element()
        element.send_keys(Keys.ALT, key)