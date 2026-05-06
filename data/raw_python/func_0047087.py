def _qr_code(self, instance):
        """
        return generate html code with "otpauth://..." link and QR-code
        """
        request = self.request # FIXME
        try:
            user = instance.user
        except ObjectDoesNotExist:
            return _("Please save first!")

        current_site = get_current_site(request)
        username = user.username
        secret = six.text_type(base64.b32encode(instance.bin_key), encoding="ASCII")

        key_uri = (
            "otpauth://totp/secure-login:%(site_name)s-%(username)s?secret=%(secret)s&issuer=%(issuer)s"
        ) % {
            "site_name": urlquote(current_site.name),
            "username": urlquote(username),
            "secret": secret,
            "issuer": urlquote(username),
        }
        context = {"key_uri": key_uri}
        return render_to_string("secure_js_login/qr_info.html", context)