def _element(cls):
        '''   find the element with controls '''
        if not cls.__is_selector():
            raise Exception("Invalid selector[%s]." %cls.__control["by"])
        
        driver = Web.driver
        try:            
            elements = WebDriverWait(driver, cls.__control["timeout"]).until(lambda driver: getattr(driver,"find_elements")(cls.__control["by"], cls.__control["value"]))
        except:                        
            raise Exception("Timeout at %d seconds.Element(%s) not found." %(cls.__control["timeout"],cls.__control["by"]))
        
        if len(elements) < cls.__control["index"] + 1:                    
            raise Exception("Element [%s]: Element Index Issue! There are [%s] Elements! Index=[%s]" % (cls.__name__, len(elements), cls.__control["index"]))
        
        if len(elements) > 1:              
            print("Element [%s]: There are [%d] elements, choosed index=%d" %(cls.__name__,len(elements),cls.__control["index"]))
        
        elm = elements[cls.__control["index"]]
        cls.__control["index"] = 0        
        return elm