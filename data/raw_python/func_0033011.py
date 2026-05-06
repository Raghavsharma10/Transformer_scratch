def child_(self, ctx):
        """
        If the root resource is requested, return the primary
        application's front page, if a primary application has been
        chosen.  Otherwise return 'self', since this page can render a
        simple index.
        """
        if self.frontPageItem.defaultApplication is None:
            return self.webViewer.wrapModel(
                _OfferingsFragment(self.frontPageItem))
        else:
            return SharingIndex(self.frontPageItem.defaultApplication.open(),
                                self.webViewer).locateChild(ctx, [''])[0]