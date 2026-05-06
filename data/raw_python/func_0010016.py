def switch_to_window(page_class, webdriver):
        """
        Utility method for switching between windows.  It will search through currently open 
        windows, then switch to the window matching the provided PageObject class.

        Args:
            page_class (PageObject): Page class to search for/instantiate.
            webdriver (WebDriver): Selenium webdriver.

        Usage::

            WebUtils.switch_to_window(DetailsPopUpPage, driver) # switches to the pop up window.

        """
        window_list = list(webdriver.window_handles)
        original_window = webdriver.current_window_handle
        for window_handle in window_list:
            webdriver.switch_to_window(window_handle)
            try:
                return PageFactory.create_page(page_class, webdriver)
            except:
                pass

        webdriver.switch_to_window(original_window)
        raise WindowNotFoundError(
            u("Window {0} not found.").format(page_class.__class__.__name__))