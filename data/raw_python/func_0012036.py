def Focus(cls):
        """        在指定输入框发送 Null， 用于设置焦点
        @note: key event ->  NULL
        """
        
        element = cls._element()        
#         element.send_keys(Keys.NULL)        
        action = ActionChains(Web.driver)
        action.send_keys_to_element(element, Keys.NULL)
        action.perform()