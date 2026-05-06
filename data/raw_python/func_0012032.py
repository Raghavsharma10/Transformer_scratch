def ReleaseClick(cls):
        ''' 释放按压操作   '''
        
        element = cls._element()        
        action = ActionChains(Web.driver)
        action.release(element)
        action.perform()