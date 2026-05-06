def Ctrl(cls, key):
        """     在指定元素上执行ctrl组合键事件
        @note: key event -> control + key
        @param key: 如'X'
        """
        element = cls._element()        
        element.send_keys(Keys.CONTROL, key)