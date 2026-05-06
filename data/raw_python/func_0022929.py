def logout(self):
        """logout func (quit browser)"""
        try:
            self.browser.quit()
        except Exception:
            raise exceptions.BrowserException(self.brow_name, "not started")
            return False
        self.vbro.stop()
        logger.info("logged out")
        return True