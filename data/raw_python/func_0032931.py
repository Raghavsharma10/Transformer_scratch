def _inject_selenium(self, test):
        """
        Injects a selenium instance into the method.
        """
        from django.conf import settings

        test_case = get_test_case_class(test)
        test_case.selenium_plugin_started = True

        # Provide some reasonable default values
        sel = selenium(
            getattr(settings, "SELENIUM_HOST", "localhost"),
            int(getattr(settings, "SELENIUM_PORT", 4444)),
            getattr(settings, "SELENIUM_BROWSER_COMMAND", "*chrome"),
            getattr(settings, "SELENIUM_URL_ROOT", "http://127.0.0.1:8000/"))

        try:
            sel.start()
        except socket.error:
            if getattr(settings, "FORCE_SELENIUM_TESTS", False):
                raise
            else:
                raise SkipTest("Selenium server not available.")
        else:
            test_case.selenium_started = True
            # Only works on method test cases, because we obviously need
            # self.
            if isinstance(test.test, nose.case.MethodTestCase):
                test.test.test.im_self.selenium = sel
            elif isinstance(test.test, TestCase):
                test.test.run.im_self.selenium = sel
            else:
                raise SkipTest("Test skipped because it's not a method.")