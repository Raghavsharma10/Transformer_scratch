def create_page(page_object_class_or_interface,
                    webdriver=None, **kwargs):
        """
        Instantiate a page object from a given Interface or Abstract class.

        Args:
            page_object_class_or_interface (Class): PageObject class, AbstractBaseClass, or 
            Interface to attempt to consturct.

        Kwargs:
            webdriver (WebDriver): Selenium Webdriver to use to instantiate the page. If none 
                                   is provided, then it was use the default from 
                                   WTF_WEBDRIVER_MANAGER

        Returns:
            PageObject

        Raises:
            NoMatchingPageError

        Instantiating a Page from PageObject from class usage::

            my_page_instance = PageFactory.create_page(MyPageClass)


        Instantiating a Page from an Interface or base class::

            import pages.mysite.*  # Make sure you import classes first, or else PageFactory will not know about it.
            my_page_instance = PageFactory.create_page(MyPageInterfaceClass)


        Instantiating a Page from a list of classes.::

            my_page_instance = PageFactory.create_page([PossiblePage1, PossiblePage2])


        Note: It'll only be able to detect pages that are imported.  To it's best to 
        do an import of all pages implementing a base class or the interface inside the 
        __init__.py of the package directory.  

        """
        if not webdriver:
            webdriver = WTF_WEBDRIVER_MANAGER.get_driver()

        # will be used later when tracking best matched page.
        current_matched_page = None

        # used to track if there is a valid page object within the set of PageObjects searched.
        was_validate_called = False

        # Walk through all classes if a list was passed.
        if type(page_object_class_or_interface) == list:
            subclasses = []
            for page_class in page_object_class_or_interface:
                # attempt to instantiate class.
                page = PageFactory.__instantiate_page_object(page_class,
                                                             webdriver,
                                                             **kwargs)

                if isinstance(page, PageObject):
                    was_validate_called = True
                    if (current_matched_page == None or page > current_matched_page):
                        current_matched_page = page

                elif page is True:
                    was_validate_called = True

                # check for subclasses
                subclasses += PageFactory.__itersubclasses(page_class)
        else:
            # A single class was passed in, try to instantiate the class.
            page_class = page_object_class_or_interface
            page = PageFactory.__instantiate_page_object(page_class,
                                                         webdriver,
                                                         **kwargs)
            # Check if we got a valid PageObject back.
            if isinstance(page, PageObject):
                was_validate_called = True
                current_matched_page = page
            elif page is True:
                was_validate_called = True

            # check for subclasses
            subclasses = PageFactory.__itersubclasses(
                page_object_class_or_interface)

        # Iterate over subclasses of the passed in classes to see if we have a
        # better match.
        for pageClass in subclasses:
            try:
                page = PageFactory.__instantiate_page_object(pageClass,
                                                      webdriver,
                                                      **kwargs)
                # If we get a valid PageObject match, check to see if the ranking is higher
                # than our current PageObject.
                if isinstance(page, PageObject):
                    was_validate_called = True
                    if current_matched_page == None or page > current_matched_page:
                        current_matched_page = page
                elif page is True:
                    was_validate_called = True

            except InvalidPageError as e:
                _wtflog.debug("InvalidPageError: %s", e)
                pass  # This happens when the page fails check.
            except TypeError as e:
                _wtflog.debug("TypeError: %s", e)
                # this happens when it tries to instantiate the original
                # abstract class.
                pass
            except Exception as e:
                _wtflog.debug("Exception during page instantiation: %s", e)
                # Unexpected exception.
                raise e

        # If no matching classes.
        if not isinstance(current_matched_page, PageObject):
            # Check that there is at least 1 valid page object that was passed in.
            if was_validate_called is False:
                raise TypeError("Neither the PageObjects nor it's subclasses have implemented " + 
                                "'PageObject._validate(self, webdriver)'.")

            try:
                current_url = webdriver.current_url
                raise NoMatchingPageError(u("There's, no matching classes to this page. URL:{0}")
                                          .format(current_url))
            except:
                raise NoMatchingPageError(u("There's, no matching classes to this page. "))
        else:
            return current_matched_page