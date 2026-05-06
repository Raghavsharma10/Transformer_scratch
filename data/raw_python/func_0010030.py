def __create_driver_from_browser_config(self):
        '''
        Reads the config value for browser type.
        '''
        try:
            browser_type = self._config_reader.get(
                WebDriverFactory.BROWSER_TYPE_CONFIG)
        except KeyError:
            _wtflog("%s missing is missing from config file. Using defaults",
                    WebDriverFactory.BROWSER_TYPE_CONFIG)
            browser_type = WebDriverFactory.FIREFOX

        # Special Chrome Sauce
        options = webdriver.ChromeOptions()
        options.add_experimental_option("excludeSwitches", ["ignore-certificate-errors"])
        options.add_argument("always-authorize-plugins")

        browser_type_dict = {
            self.CHROME: lambda: webdriver.Chrome(
                self._config_reader.get(WebDriverFactory.CHROME_DRIVER_PATH),
                chrome_options=options),
            self.FIREFOX: lambda: webdriver.Firefox(),
            self.INTERNETEXPLORER: lambda: webdriver.Ie(),
            self.OPERA: lambda: webdriver.Opera(),
            self.PHANTOMJS: lambda: self.__create_phantom_js_driver(),
            self.SAFARI: lambda: self.__create_safari_driver()
        }

        try:
            return browser_type_dict[browser_type]()
        except KeyError:
            raise TypeError(
                u("Unsupported Browser Type {0}").format(browser_type))