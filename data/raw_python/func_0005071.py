def get_isogeo_version(self, component: str = "api", prot: str = "https"):
        """Get Isogeo components versions. Authentication not required.

        :param str component: which platform component. Options:

          * api [default]
          * db
          * app

        """
        # which component
        if component == "api":
            version_url = "{}://v1.{}.isogeo.com/about".format(prot, self.api_url)
        elif component == "db":
            version_url = "{}://v1.{}.isogeo.com/about/database".format(
                prot, self.api_url
            )
        elif component == "app" and self.platform == "prod":
            version_url = "https://app.isogeo.com/about"
        elif component == "app" and self.platform == "qa":
            version_url = "https://qa-isogeo-app.azurewebsites.net/about"
        else:
            raise ValueError(
                "Component value must be one of: " "api [default], db, app."
            )

        # send request
        version_req = requests.get(version_url, proxies=self.proxies, verify=self.ssl)

        # checking response
        checker.check_api_response(version_req)

        # end of method
        return version_req.json().get("version")