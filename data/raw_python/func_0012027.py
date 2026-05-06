def Click(cls):
        ''' 左键 点击 1次   '''
        
        element= cls._element()        
        action = ActionChains(Web.driver)
        action.click(element)
        action.perform()