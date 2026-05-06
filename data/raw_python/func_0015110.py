def bm_create_button(self, **kwargs):
        """Shortcut to the BMCreateButton method.

        See the docs for details on arguments:
        https://cms.paypal.com/mx/cgi-bin/?cmd=_render-content&content_ID=developer/e_howto_api_nvp_BMCreateButton

        The L_BUTTONVARn fields are especially important, so make sure to
        read those and act accordingly. See unit tests for some examples.
        """
        kwargs.update(self._sanitize_locals(locals()))
        return self._call('BMCreateButton', **kwargs)