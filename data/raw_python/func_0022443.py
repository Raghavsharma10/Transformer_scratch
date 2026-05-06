def connect_to_wifi(self, ssid, password=None):
        """
        [Test Agent]

        Connect to *ssid* with *password*
        """
        cmd = 'am broadcast -a testagent -e action CONNECT_TO_WIFI -e ssid %s -e password %s' % (ssid, password)
        self.adb.shell_cmd(cmd)