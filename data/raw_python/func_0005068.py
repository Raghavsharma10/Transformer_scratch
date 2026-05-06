def set_base_url(self, platform: str = "prod"):
        """Set Isogeo base URLs according to platform.

        :param str platform: platform to use. Options:

          * prod [DEFAULT]
          * qa
          * int
        """
        platform = platform.lower()
        self.platform = platform
        if platform == "prod":
            ssl = True
            logging.debug("Using production platform.")
        elif platform == "qa":
            ssl = False
            logging.debug("Using Quality Assurance platform (reduced perfs).")
        else:
            logging.error(
                "Platform must be one of: {}".format(" | ".join(self.API_URLS.keys()))
            )
            raise ValueError(
                3,
                "Platform must be one of: {}".format(" | ".join(self.API_URLS.keys())),
            )
        # method ending
        return (
            platform.lower(),
            self.API_URLS.get(platform),
            self.APP_URLS.get(platform),
            self.CSW_URLS.get(platform),
            self.MNG_URLS.get(platform),
            self.OC_URLS.get(platform),
            ssl,
        )