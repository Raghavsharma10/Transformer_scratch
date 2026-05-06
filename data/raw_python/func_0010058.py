def wait_until_element_not_visible(webdriver, locator_lambda_expression,
                                       timeout=WTF_TIMEOUT_MANAGER.NORMAL, sleep=0.5):
        """
        Wait for a WebElement to disappear.

        Args:
            webdriver (Webdriver) - Selenium Webdriver
            locator (lambda) - Locator lambda expression.

        Kwargs:
            timeout (number) - timeout period
            sleep (number) - sleep period between intervals.

        """
        # Wait for loading progress indicator to go away.
        try:
            stoptime = datetime.now() + timedelta(seconds=timeout)
            while datetime.now() < stoptime:
                element = WebDriverWait(webdriver, WTF_TIMEOUT_MANAGER.BRIEF).until(
                    locator_lambda_expression)
                if element.is_displayed():
                    time.sleep(sleep)
                else:
                    break
        except TimeoutException:
            pass