def RightClick(cls):
        ''' 右键点击1次 '''
        
        element = cls._element()        
        action = ActionChains(Web.driver)
        action.context_click(element)
        action.perform()