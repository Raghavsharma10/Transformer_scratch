def launch(self):
        """launch browser and virtual display, first of all to be launched"""
        try:
            # init virtual Display
            self.vbro = Display()
            self.vbro.start()
            logger.debug("virtual display launched")
        except Exception:
            raise exceptions.VBroException()
        try:
            self.browser = Browser(self.brow_name)
            logger.debug(f"browser {self.brow_name} launched")
        except Exception:
            raise exceptions.BrowserException(
                self.brow_name, "failed to launch")
        return True