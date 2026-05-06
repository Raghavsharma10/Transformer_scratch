def qr_code(self, instance):
        """
        Display picture of QR-code from used secret
        """
        try:
            return self._qr_code(instance)
        except Exception as err:
            if settings.DEBUG:
                import traceback
                return "<pre>%s</pre>" % traceback.format_exc()