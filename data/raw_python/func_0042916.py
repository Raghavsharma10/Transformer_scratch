def history(self):
        """ History """
        logger.info("Transactions:")
        try:
            logger.debug("Account: %s", self.account)
            self.request("engagement/transactions/%s" % self.account)
        except HTTPError as e:
            error = json.loads(e.read().decode("utf8"))
            logger.error(error["errorMessages"]["general"][0]["message"])
            return

        transactions = json.loads(self.getdata())["transactions"]
        ret = list()
        for i in transactions:
            ret.append([i["date"], i["description"], i["amount"]])
        return ret