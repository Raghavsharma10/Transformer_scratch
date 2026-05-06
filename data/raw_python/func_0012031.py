def ClickAndHold(cls):
        ''' 相当于 按压，press '''
        
        element = cls._element()        
        action = ActionChains(Web.driver)
        action.click_and_hold(element)
        action.perform()