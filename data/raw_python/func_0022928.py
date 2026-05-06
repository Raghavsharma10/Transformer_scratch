def login(self, username, password, mode="demo"):
        """login function"""
        url = "https://trading212.com/it/login"
        try:
            logger.debug(f"visiting %s" % url)
            self.browser.visit(url)
            logger.debug(f"connected to %s" % url)
        except selenium.common.exceptions.WebDriverException:
            logger.critical("connection timed out")
            raise
        try:
            self.search_name("login[username]").fill(username)
            self.search_name("login[password]").fill(password)
            self.css1(path['log']).click()
            # define a timeout for logging in
            timeout = time.time() + 30
            while not self.elCss(path['logo']):
                if time.time() > timeout:
                    logger.critical("login failed")
                    raise CredentialsException(username)
            time.sleep(1)
            logger.info(f"logged in as {username}")
            # check if it's a weekend
            if mode == "demo" and datetime.now().isoweekday() in range(5, 8):
                timeout = time.time() + 10
                while not self.elCss(path['alert-box']):
                    if time.time() > timeout:
                        logger.warning("weekend trading alert-box not closed")
                        break
                if self.elCss(path['alert-box']):
                    self.css1(path['alert-box']).click()
                    logger.debug("weekend trading alert-box closed")
        except Exception as e:
            logger.critical("login failed")
            raise exceptions.BaseExc(e)
        return True