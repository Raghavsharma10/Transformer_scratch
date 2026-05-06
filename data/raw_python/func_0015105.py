def generate_express_checkout_redirect_url(self, token, useraction=None):
        """Returns the URL to redirect the user to for the Express checkout.

        Express Checkouts must be verified by the customer by redirecting them
        to the PayPal website. Use the token returned in the response from
        :meth:`set_express_checkout` with this function to figure out where
        to redirect the user to.

        The button text on the PayPal page can be controlled via `useraction`.
        The documented possible values are `commit` and `continue`. However,
        any other value will only result in a warning.

        :param str token: The unique token identifying this transaction.
        :param str useraction: Control the button text on the PayPal page.
        :rtype: str
        :returns: The URL to redirect the user to for approval.
        """
        url_vars = (self.config.PAYPAL_URL_BASE, token)
        url = "%s?cmd=_express-checkout&token=%s" % url_vars
        if useraction:
            if not useraction.lower() in ('commit', 'continue'):
                warnings.warn('useraction=%s is not documented' % useraction,
                              RuntimeWarning)
            url += '&useraction=%s' % useraction
        return url