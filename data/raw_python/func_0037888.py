def send_token_email(self, request, queryset):
        """
        Sends token email(s) for the selected users.

        """
        for token in queryset:
            if not token.expired: forward_token(token)