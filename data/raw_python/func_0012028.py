def DoubleClick(cls):
        ''' 左键点击2次 '''
        
        element = cls._element()        
        action = ActionChains(Web.driver)
        action.double_click(element)
        action.perform()