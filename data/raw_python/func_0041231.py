def _run__hook(self, action, replace):
        """Simple webhook"""
        url = action.get("url")
        expected = action.get("expect", {}).get("response-codes", (200, 201, 202, 204))
        if replace and action.get("template", True):
            url = self.rfxcfg.macro_expand(url, replace)
        self.logf("Action {} hook\n", action['name'])
        self.logf("{}\n", url, level=common.log_msg)
        result = requests.get(url)
        self.debug("Result={}\n", result.status_code)
        if result.status_code not in expected:
            self.die("Hook failed name={} result={}", action['name'], result.status_code)
        self.logf("Success\n", level=common.log_good)