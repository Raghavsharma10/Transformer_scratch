def get_logs(self):
        """
        Return the system logs since the last reboot.
        """
        assert BeautifulSoup, "Please install bs4 to use this method"

        url = self.base_url + "/system/syslog.lua"
        response = self.session.get(url, params={
            'sid': self.sid,
            'stylemode': 'print',
        }, timeout=15)
        response.raise_for_status()

        entries = []
        tree = BeautifulSoup(response.text)
        rows = tree.find('table').find_all('tr')
        for row in rows:
            columns = row.find_all("td")
            date = columns[0].string
            time = columns[1].string
            message = columns[2].find("a").string

            merged = "{} {} {}".format(date, time, message.encode("UTF-8"))
            msg_hash = hashlib.md5(merged).hexdigest()
            entries.append(LogEntry(date, time, message, msg_hash))
        return entries