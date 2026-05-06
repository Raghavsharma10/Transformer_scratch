def __instantiate_page_object(page_obj_class, webdriver, **kwargs):
        """
        Attempts to instantiate a page object.

        Args:
            page_obj_class (PageObject) - PageObject to instantiate.
            webdriver (WebDriver) - Selenium webdriver to associate with the PageObject
        
        Returns:
            PageObject - If page object instantiation succeeded.
            True - If page object instantiation failed, but validation was called.
            None - If validation did not occur.

        """
        try:
            page = page_obj_class(webdriver, **kwargs)
            return page
        except InvalidPageError:
            # This happens when the page fails check.
            # Means validate was implemented, but the check didn't pass.
            return True
        except TypeError:
            # this happens when it tries to instantiate the original abstract
            # class, or a PageObject where _validate() was not implemented.
            return False
        except Exception as e:
            # Unexpected exception.
            raise e