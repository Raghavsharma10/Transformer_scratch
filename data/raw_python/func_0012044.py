def is_runnnig(self):
        """Determine whether hub server is running
        :return:True or False
        """
        resp = None
        try:
            resp = requests.get("http://%s:%s" %(self._ip, self._port))

            if resp.status_code == 200:
                return True
            else:
                return False
        except:
            return False