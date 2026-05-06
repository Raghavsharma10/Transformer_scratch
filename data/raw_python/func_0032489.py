def postOptions(self):
        """
        Find an installed offering and set the site front page to its
        application's front page.
        """
        o = self.store.findFirst(
            offering.InstalledOffering,
            (offering.InstalledOffering.offeringName ==
             self["name"]))
        if o is None:
            raise usage.UsageError("No offering of that name"
                                   " is installed.")
        fp = self.store.findUnique(publicweb.FrontPage)
        fp.defaultApplication = o.application