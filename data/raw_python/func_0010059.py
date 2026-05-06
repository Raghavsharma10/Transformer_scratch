def is_image_loaded(webdriver, webelement):
        '''
        Check if an image (in an image tag) is loaded.
        Note: This call will not work against background images.  Only Images in <img> tags.

        Args:
            webelement (WebElement) - WebDriver web element to validate.

        '''
        script = (u("return arguments[0].complete && type of arguments[0].naturalWidth != \"undefined\" ") + 
                 u("&& arguments[0].naturalWidth > 0"))
        try:
            return webdriver.execute_script(script, webelement)
        except:
            return False