def installSite(self, siteStore, domain, publicURL, generateCert=True):
        """
        Create the necessary items to run an HTTP server and an SSH server.
        """
        certPath = siteStore.filesdir.child("server.pem")
        if generateCert and not certPath.exists():
            certPath.setContent(self._createCert(domain, genSerial()))

        # Install the base Mantissa offering.
        IOfferingTechnician(siteStore).installOffering(baseOffering)

        # Make the HTTP server baseOffering includes listen somewhere.
        site = siteStore.findUnique(SiteConfiguration)
        site.hostname = domain
        installOn(
            TCPPort(store=siteStore, factory=site, portNumber=8080),
            siteStore)
        installOn(
            SSLPort(store=siteStore, factory=site, portNumber=8443,
                    certificatePath=certPath),
            siteStore)

        # Make the SSH server baseOffering includes listen somewhere.
        shell = siteStore.findUnique(SecureShellConfiguration)
        installOn(
            TCPPort(store=siteStore, factory=shell, portNumber=8022),
            siteStore)

        # Install a front page on the top level store so that the
        # developer will have something to look at when they start up
        # the server.
        fp = siteStore.findOrCreate(publicweb.FrontPage, prefixURL=u'')
        installOn(fp, siteStore)