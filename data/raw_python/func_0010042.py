def take_reference_screenshot(webdriver, file_name):
        """
        Captures a screenshot as a reference screenshot.

        Args:
            webdriver (WebDriver) - Selenium webdriver.
            file_name (str) - File name to save screenshot as.
        """
        folder_location = os.path.join(ProjectUtils.get_project_root(),
                                       WebScreenShotUtil.REFERENCE_SCREEN_SHOT_LOCATION)

        WebScreenShotUtil.__capture_screenshot(
            webdriver, folder_location, file_name + ".png")