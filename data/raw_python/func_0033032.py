def installOffering(self, offering):
        """
        Install the given offering::

          - Create and install the powerups in its I{siteRequirements} list.
          - Create an application L{Store} and a L{LoginAccount} referring to
            it.  Install the I{appPowerups} on the application store.
          - Create an L{InstalledOffering.

        Perform all of these tasks in a transaction managed within the scope of
        this call (that means you should not call this function inside a
        transaction, or you should not handle any exceptions it raises inside
        an externally managed transaction).

        @type offering: L{IOffering}
        @param offering: The offering to install.

        @return: The C{InstalledOffering} item created.
        """
        for off in self._siteStore.query(
            InstalledOffering,
            InstalledOffering.offeringName == offering.name):
            raise OfferingAlreadyInstalled(off)

        def siteSetup():
            for (requiredInterface, requiredPowerup) in offering.siteRequirements:
                if requiredInterface is not None:
                    nn = requiredInterface(self._siteStore, None)
                    if nn is not None:
                        continue
                if requiredPowerup is None:
                    raise NotImplementedError(
                        'Interface %r required by %r but not provided by %r' %
                        (requiredInterface, offering, self._siteStore))
                self._siteStore.findOrCreate(
                    requiredPowerup, lambda p: installOn(p, self._siteStore))

            ls = self._siteStore.findOrCreate(userbase.LoginSystem)
            substoreItem = substore.SubStore.createNew(
                self._siteStore, ('app', offering.name + '.axiom'))
            ls.addAccount(offering.name, None, None, internal=True,
                          avatars=substoreItem)

            from xmantissa.publicweb import PublicWeb
            PublicWeb(store=self._siteStore, application=substoreItem,
                      prefixURL=offering.name)
            ss = substoreItem.open()
            def appSetup():
                for pup in offering.appPowerups:
                    installOn(pup(store=ss), ss)

            ss.transact(appSetup)
            # Woops, we need atomic cross-store transactions.
            io = InstalledOffering(
                store=self._siteStore, offeringName=offering.name,
                application=substoreItem)

            #Some new themes may be available now. Clear the theme cache
            #so they can show up.
            #XXX This is pretty terrible -- there
            #really should be a scheme by which ThemeCache instances can
            #be non-global. Fix this at the earliest opportunity.
            from xmantissa import webtheme
            webtheme.theThemeCache.emptyCache()
            return io
        return self._siteStore.transact(siteSetup)