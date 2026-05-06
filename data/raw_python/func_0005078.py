def credentials_loader(self, in_credentials: str = "client_secrets.json") -> dict:
        """Loads API credentials from a file, JSON or INI.

        :param str in_credentials: path to the credentials file. By default,
          look for a client_secrets.json file.
        """
        accepted_extensions = (".ini", ".json")
        # checks
        if not path.isfile(in_credentials):
            raise IOError("Credentials file doesn't exist: {}".format(in_credentials))
        else:
            in_credentials = path.normpath(in_credentials)
        if path.splitext(in_credentials)[1] not in accepted_extensions:
            raise ValueError(
                "Extension of credentials file must be one of {}".format(
                    accepted_extensions
                )
            )
        else:
            kind = path.splitext(in_credentials)[1]
        # load, check and set
        if kind == ".json":
            with open(in_credentials, "r") as f:
                in_auth = json.loads(f.read())
            # check structure
            heads = ("installed", "web")
            if not set(in_auth).intersection(set(heads)):
                raise ValueError(
                    "Input JSON structure is not as expected."
                    " First key must be one of: {}".format(heads)
                )
            # set
            if "web" in in_auth:
                # json structure for group application
                auth_settings = in_auth.get("web")
                out_auth = {
                    "auth_mode": "group",
                    "client_id": auth_settings.get("client_id"),
                    "client_secret": auth_settings.get("client_secret"),
                    # if not specified, must be a former file then set classic scope
                    "scopes": auth_settings.get("scopes", ["resources:read"]),
                    "uri_auth": auth_settings.get("auth_uri"),
                    "uri_token": auth_settings.get("token_uri"),
                    "uri_base": self.get_url_base_from_url_token(
                        auth_settings.get("token_uri")
                    ),
                    "uri_redirect": None,
                }
            else:
                # assuming in_auth == 'installed'
                auth_settings = in_auth.get("installed")
                out_auth = {
                    "auth_mode": "user",
                    "client_id": auth_settings.get("client_id"),
                    "client_secret": auth_settings.get("client_secret"),
                    # if not specified, must be a former file then set classic scope
                    "scopes": auth_settings.get("scopes", ["resources:read"]),
                    "uri_auth": auth_settings.get("auth_uri"),
                    "uri_token": auth_settings.get("token_uri"),
                    "uri_base": self.get_url_base_from_url_token(
                        auth_settings.get("token_uri")
                    ),
                    "uri_redirect": auth_settings.get("redirect_uris", None),
                }
        else:
            # assuming file is an .ini
            ini_parser = ConfigParser()
            ini_parser.read(in_credentials)
            # check structure
            if "auth" in ini_parser._sections:
                auth_settings = ini_parser["auth"]
            else:
                raise ValueError(
                    "Input INI structure is not as expected."
                    " Section of credentials must be named: auth"
                )
            # set
            out_auth = {
                "auth_mode": auth_settings.get("CLIENT_TYPE"),
                "client_id": auth_settings.get("CLIENT_ID"),
                "client_secret": auth_settings.get("CLIENT_SECRET"),
                "uri_auth": auth_settings.get("URI_AUTH"),
                "uri_token": auth_settings.get("URI_TOKEN"),
                "uri_base": self.get_url_base_from_url_token(
                    auth_settings.get("URI_TOKEN")
                ),
                "uri_redirect": auth_settings.get("URI_REDIRECT"),
            }
        # method ending
        return out_auth