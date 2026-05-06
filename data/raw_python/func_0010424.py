def inject(self):
        """
        Recursively inject aXe into all iframes and the top level document.

        :param script_url: location of the axe-core script.
        :type script_url: string
        """
        with open(self.script_url, "r", encoding="utf8") as f:
            self.selenium.execute_script(f.read())