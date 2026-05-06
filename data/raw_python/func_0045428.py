def authenticate(identity=None, provider=None):
        " Authenticate user by net identity. "
        if not identity:
            return None

        try:
            netid = NetID.objects.get(identity=identity, provider=provider)
            return netid.user
        except NetID.DoesNotExist:
            return None