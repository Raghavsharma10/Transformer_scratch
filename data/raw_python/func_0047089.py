def tokens(self, instance):
        """
        Just display current acceptable TOTP tokens
        """
        if not instance.pk:
            # e.g.: Use will create a new TOTP entry
            return "-"

        totp = TOTP(instance.bin_key, instance.step, instance.t0, instance.digits)

        tokens = []
        for offset in range(-instance.tolerance, instance.tolerance + 1):
            totp.drift = instance.drift + offset
            tokens.append(totp.token())

        return " ".join(["%s" % token for token in tokens])