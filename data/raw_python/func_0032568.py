def rootChild_resetPassword(self, req, webViewer):
        """
        Redirect authenticated users to their settings page (hopefully they
        have one) when they try to reset their password.

        This is the wrong way for this functionality to be implemented.  See
        #2524.
        """
        from xmantissa.ixmantissa import IWebTranslator, IPreferenceAggregator
        return URL.fromString(
            IWebTranslator(self.store).linkTo(
                IPreferenceAggregator(self.store).storeID))