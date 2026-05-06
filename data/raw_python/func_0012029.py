def EnhancedClick(cls):
        '''
        Description:
            Sometimes, one click on the element doesn't work. So wait more time, then click again and again.
        Risk:
            It may operate more than one click operations.
        '''
        
        element = cls._element()
        for _ in range(3):
            action = ActionChains(Web.driver)
            action.move_to_element(element)
            action.perform()           
            time.sleep(0.5)