def accounts(self):
        """ Accounts """
        logger.info("Fetching data...")
        try:
            self.request("engagement/overview")
        except HTTPError as e:
            error = json.loads(e.read().decode("utf8"))
            logger.error(error["errorMessages"]["general"][0]["message"])
            return
        overview = json.loads(self.getdata())
        overviewl = reversed(list(overview))
        ret = list()
        for i in overviewl:
            if len(overview[i]) > 0:
                for n in overview[i]:
                    if self.account is None and "id" in n:
                        self.account = n["id"]
                    if n.get('balance'):
                        ret.append({n['name']: n['balance']})
                    elif n.get('availableAmount', None):
                        ret.append({n['name']: n['availableAmount']})

                    else:
                        logger.error("Unable to parse %s", n)
        return ret