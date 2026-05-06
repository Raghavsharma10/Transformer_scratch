def get_view_url(self, webapp: str = "oc", **kwargs):
        """Constructs the view URL of a metadata.

        :param str webapp: web app destination
        :param dict kwargs: web app specific parameters. For example see WEBAPPS
        """
        # build wbeapp URL depending on choosen webapp
        if webapp in self.WEBAPPS:
            webapp_args = self.WEBAPPS.get(webapp).get("args")
            # check kwargs parameters
            if set(webapp_args) <= set(kwargs):
                # construct and return url
                url = self.WEBAPPS.get(webapp).get("url")

                return url.format(**kwargs)
            else:
                raise TypeError(
                    "'{}' webapp expects {} argument(s): {}."
                    " Args passed: {}".format(
                        webapp, len(webapp_args), webapp_args, kwargs
                    )
                )
        else:
            raise ValueError(
                "'{}' is not a recognized webapp among: {}."
                " Try to register it.".format(self.WEBAPPS.keys(), webapp)
            )