def _log_fail_callback(driver, *args, **kwargs):
    """Raises an assertion error if the page has severe console errors

    @param driver: ShapewaysDriver
    @return: None
    """

    try:
        logs = driver.get_browser_log(levels=[BROWSER_LOG_LEVEL_SEVERE])
        failure_message = 'There were severe console errors on this page: {}'.format(logs)
        failure_message = failure_message.replace('{', '{{').replace('}', '}}')  # Escape braces for error message
        driver.assertion.assert_false(
            logs,
            failure_message=failure_message
        )
    except (urllib2.URLError, socket.error, WebDriverException):
        # The session has ended, don't check the logs
        pass