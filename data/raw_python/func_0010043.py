def __capture_screenshot(webdriver, folder_location, file_name):
        "Capture a screenshot"
        # Check folder location exists.
        if not os.path.exists(folder_location):
            os.makedirs(folder_location)

        file_location = os.path.join(folder_location, file_name)

        if isinstance(webdriver, remote.webdriver.WebDriver):
            # If this is a remote webdriver.  We need to transmit the image data
            # back across system boundries as a base 64 encoded string so it can
            # be decoded back on the local system and written to disk.
            base64_data = webdriver.get_screenshot_as_base64()
            screenshot_data = base64.decodestring(base64_data)
            screenshot_file = open(file_location, "wb")
            screenshot_file.write(screenshot_data)
            screenshot_file.close()
        else:
            webdriver.save_screenshot(file_location)