def wait_until_page_ready(page_object, timeout=WTF_TIMEOUT_MANAGER.NORMAL):
        """Waits until document.readyState == Complete (e.g. ready to execute javascript commands)

        Args:
            page_object (PageObject) : PageObject class

        Kwargs:
            timeout (number) : timeout period
        """
        try:
            do_until(lambda: page_object.webdriver.execute_script("return document.readyState").lower()
                     == 'complete', timeout)
        except wait_utils.OperationTimeoutError:
            raise PageUtilOperationTimeoutError(
                "Timeout occurred while waiting for page to be ready.")