def cred_init(
            self,
            *,
            secrets_dir: str,
            log: Logger,
            bot_name: str="",
    ) -> None:
        """Initialize what requires credentials/secret files."""
        super().__init__(secrets_dir=secrets_dir, log=log, bot_name=bot_name)

        self.ldebug("Retrieving ACCESS_TOKEN ...")
        with open(path.join(self.secrets_dir, "ACCESS_TOKEN")) as f:
            ACCESS_TOKEN = f.read().strip()

        # Instance base url optional.
        self.ldebug("Looking for INSTANCE_BASE_URL ...")
        instance_base_url_path = path.join(self.secrets_dir, "INSTANCE_BASE_URL")
        if path.isfile(instance_base_url_path):
            with open(instance_base_url_path) as f:
                self.instance_base_url = f.read().strip()
        else:
            self.ldebug("Couldn't find INSTANCE_BASE_URL, defaulting to mastodon.social.")
            self.instance_base_url = "https://mastodon.social"

        self.api = mastodon.Mastodon(access_token=ACCESS_TOKEN,
                                     api_base_url=self.instance_base_url)
        self.html_re = re.compile("<.*?>")