def _elements(cls):
        '''   find the elements with controls '''
        if not cls.__is_selector():
            raise Exception("Invalid selector[%s]." %cls.__control["by"])
        
        driver = Web.driver
        try:            
            elements = WebDriverWait(driver, cls.__control["timeout"]).until(lambda driver: getattr(driver,"find_elements")(cls.__control["by"], cls.__control["value"]))
        except:            
            raise Exception("Timeout at %d seconds.Element(%s) not found." %(cls.__control["timeout"],cls.__control["by"]))
            
        return elements