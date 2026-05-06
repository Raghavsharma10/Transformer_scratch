def cred_init(
            self,
            *,
            secrets_dir: str,
            log: Logger,
            bot_name: str,
    ) -> None:
        """
        Initialize what requires credentials/secret files.

        :param secrets_dir: dir to expect credentials in and store logs/history in.
        :param log: logger to use for log output.
        :param bot_name: name of this bot,
            used for various kinds of labelling.
        :returns: none.
        """
        super().__init__(secrets_dir=secrets_dir, log=log, bot_name=bot_name)

        self.ldebug("Retrieving CONSUMER_KEY...")
        with open(path.join(self.secrets_dir, "CONSUMER_KEY")) as f:
            CONSUMER_KEY = f.read().strip()

        self.ldebug("Retrieving CONSUMER_SECRET...")
        with open(path.join(self.secrets_dir, "CONSUMER_SECRET")) as f:
            CONSUMER_SECRET = f.read().strip()

        self.ldebug("Retrieving ACCESS_TOKEN...")
        with open(path.join(self.secrets_dir, "ACCESS_TOKEN")) as f:
            ACCESS_TOKEN = f.read().strip()

        self.ldebug("Retrieving ACCESS_SECRET...")
        with open(path.join(self.secrets_dir, "ACCESS_SECRET")) as f:
            ACCESS_SECRET = f.read().strip()

        self.ldebug("Looking for OWNER_HANDLE...")
        owner_handle_path = path.join(self.secrets_dir, "OWNER_HANDLE")
        if path.isfile(owner_handle_path):
            with open(owner_handle_path) as f:
                self.owner_handle = f.read().strip()
        else:
            self.ldebug("Couldn't find OWNER_HANDLE, unable to DM...")
            self.owner_handle = ""

        self.auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        self.auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

        self.api = tweepy.API(self.auth)