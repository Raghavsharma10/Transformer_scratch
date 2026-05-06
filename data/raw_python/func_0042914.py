def login(self, user, passwd, bank):
        """ Login """
        logger.info("login...")
        if bank not in self.BANKS:
            logger.error("Can't find that bank.")
            return False
        self.useragent = self.BANKS[bank]["u-a"]
        self.bankid = self.BANKS[bank]["id"]
        login = json.dumps(
                {"userId": user, "password": passwd, "useEasyLogin": False,
                 "generateEasyLoginId": False})
        try:
            self.request("identification/personalcode", post=login,
                         method="POST")
        except HTTPError as e:
            error = json.loads(e.read().decode("utf8"))
            logger.error(error["errorMessages"]["fields"][0]["message"])
            return False
        try:
            self.request("profile/")
        except HTTPError as e:
            error = json.loads(e.read().decode("utf8"))
            logger.error(error["errorMessages"]["general"][0]["message"])
            return False

        profile = json.loads(self.getdata())
        if len(profile["banks"]) == 0:
            logger.error("Using wrong bank? Can't find any bank info.")
            return False
        try:
            self.profile = profile["banks"][0]["privateProfile"]["id"]
        except KeyError:
            self.profile = profile['banks'][0]['corporateProfiles'][0]["id"]
        try:
            self.request("profile/%s" % self.profile, method="POST")
        except HTTPError as e:
            error = json.loads(e.read().decode("utf8"))
            logger.error(error["errorMessages"]["general"][0]["message"])
            return False

        return True