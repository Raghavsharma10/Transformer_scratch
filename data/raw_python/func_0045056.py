def merge_accounts(self, request):
        """
        Attach NetID account to regular django
        account and then redirect user. In this situation
        user dont have to fill extra fields because he filled
        them when first account (request.user) was created.

        Note that self.indentity must be already set in this stage by
        validate_response function.
        """
        # create new net ID record in database
        # and attach it to request.user account.
        try:
            netid = NetID.objects.get(identity=self.identity, provider=self.provider)
        except NetID.DoesNotExist:
            netid = NetID(user=request.user, identity=self.identity, provider=self.provider)
            netid.save()
            # show nice message to user.
            messages.add_message(request, messages.SUCCESS, lang.ACCOUNTS_MERGED)