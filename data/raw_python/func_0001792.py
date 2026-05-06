def _get_auth(self, user=None, password=None):
        """
        Get the authentication info for the current user, from
        1) a ~/.amcatauth file, which should be a csv file
          containing host, username, password entries
        2) the AMCAT_USER (or USER) and AMCAT_PASSWORD environment variables
        """
        fn = os.path.expanduser(AUTH_FILE)
        if os.path.exists(fn):
            for i, line in enumerate(csv.reader(open(fn))):
                if len(line) != 3:
                    log.warning("Cannot parse line {i} in {fn}".format(**locals()))
                    continue
                hostname, username, pwd = line
                if (hostname in ("", "*", self.host)
                    and (user is None or username == user)):
                    return (username, pwd)
        if user is None:
            user = os.environ.get("AMCAT_USER", os.environ.get("USER"))
        if password is None:
            password = os.environ.get("AMCAT_PASSWORD")
        if user is None or password is None:
            raise Exception("No authentication info for {user}@{self.host} "
                            "from {fn} or AMCAT_USER / AMCAT_PASSWORD "
                            "variables".format(**locals()))
        return user, password