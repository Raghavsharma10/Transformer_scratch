def Enter(cls):
        '''     在指定输入框发送回回车键
        @note: key event -> enter
        '''
        
        element = cls._element()        
        action = ActionChains(Web.driver)
        action.send_keys_to_element(element, Keys.ENTER)
        action.perform()