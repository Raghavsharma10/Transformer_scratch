def pause_and_wait_for_user(self, timeout=None, prompt_text='Click to resume (WebDriver is paused)'):
        """Injects a radio button into the page and waits for the user to click it; will raise an exception if the
        radio to resume is never checked

        @return: None
        """
        timeout = timeout if timeout is not None else self.user_wait_timeout
        # Set the browser state paused
        self.paused = True

        def check_user_ready(driver):
            """Polls for the user to be "ready" (meaning they checked the checkbox) and the driver to be unpaused.
            If the checkbox is not displayed (e.g. user navigates the page), it will re-insert it into the page

            @type driver: WebDriverWrapper
            @param driver: Driver to execute
            @return: True if user is ready, false if not
            """
            if driver.paused:
                if driver.is_user_ready():
                    # User indicated they are ready; free the browser lock
                    driver.paused = False
                    return True
                else:
                    if not driver.is_present(Locator('css', '#webdriver-resume-radio', 'radio to unpause webdriver')):
                        # Display the prompt
                        pause_html = staticreader.read_html_file('webdriverpaused.html')\
                            .replace('\n', '')\
                            .replace('PROMPT_TEXT', prompt_text)
                        webdriver_style = staticreader.read_css_file('webdriverstyle.css').replace('\n', '')


                        # Insert the webdriver style
                        driver.js_executor.execute_template_and_return_result(
                            'injectCssTemplate.js',
                            {'css': webdriver_style})

                        # Insert the paused html
                        driver.js_executor.execute_template_and_return_result(
                            'injectHtmlTemplate.js',
                            {'selector': 'body', 'html': pause_html})
            return False

        self.wait_until(
            lambda: check_user_ready(self),
            timeout=timeout,
            failure_message='Webdriver actions were paused but did not receive the command to continue. '
                            'You must click the on-screen message to resume.'
        )

        # Remove all injected elements
        self.js_executor.execute_template_and_return_result(
            'deleteElementsTemplate.js',
            {'selector': '.webdriver-injected'}
        )