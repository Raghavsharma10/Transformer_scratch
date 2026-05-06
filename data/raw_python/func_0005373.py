def get_app_properties(self, token: dict = None, prot: str = "https"):
        """Get information about the application declared on Isogeo.

        :param str token: API auth token
        :param str prot: https [DEFAULT] or http
         (use it only for dev and tracking needs).
        """
        # check if app properties have already been retrieved or not
        if not hasattr(self, "app_properties"):
            first_app = self.shares()[0].get("applications")[0]
            app = {
                "admin_url": "{}/applications/{}".format(
                    self.mng_url, first_app.get("_id")
                ),
                "creation_date": first_app.get("_created"),
                "last_update": first_app.get("_modified"),
                "name": first_app.get("name"),
                "type": first_app.get("type"),
                "kind": first_app.get("kind"),
                "url": first_app.get("url"),
            }
            self.app_properties = app
        else:
            pass