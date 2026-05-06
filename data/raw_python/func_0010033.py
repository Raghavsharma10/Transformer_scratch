def __create_remote_webdriver_from_config(self, testname=None):
        '''
        Reads the config value for browser type.
        '''
        desired_capabilities = self._generate_desired_capabilities(testname)
        
        remote_url = self._config_reader.get(
            WebDriverFactory.REMOTE_URL_CONFIG)

        # Instantiate remote webdriver.
        driver = webdriver.Remote(
            desired_capabilities=desired_capabilities,
            command_executor=remote_url
        )

        # Log IP Address of node if configured, so it can be used to
        # troubleshoot issues if they occur.
        log_driver_props = \
            self._config_reader.get(
                WebDriverFactory.LOG_REMOTEDRIVER_PROPS, default_value=False
            ) in [True, "true", "TRUE", "True"]
        if "wd/hub" in remote_url and log_driver_props:
            try:
                grid_addr = remote_url[:remote_url.index("wd/hub")]
                info_request_response = urllib2.urlopen(
                    grid_addr + "grid/api/testsession?session=" + driver.session_id, "", 5000)
                node_info = info_request_response.read()
                _wtflog.info(
                    u("RemoteWebdriver using node: ") + u(node_info).strip())
            except:
                # Unable to get IP Address of remote webdriver.
                # This happens with many 3rd party grid providers as they don't want you accessing info on nodes on
                # their internal network.
                pass

        return driver