def _determine_auth_mechanism(username, password, delegation):
        """
        if the username contains at '@' sign we will use kerberos
        if the username contains a '/ we will use ntlm
        either NTLM or Kerberos. In fact its basically always Negotiate.
        """
        if re.match('(.*)@(.+)', username) is not None:
            if delegation is True:
                raise Exception('Kerberos is not yet supported, specify the username in <domain>\<username> form for NTLM')
            else:
                raise Exception('Kerberos is not yet supported, specify the username in <domain>>\<username> form for NTLM')

        # check for NT format 'domain\username' a blank domain or username is invalid
        legacy = re.match('(.*)\\\\(.*)', username)
        if legacy is not None:
            if not legacy.group(1):
                raise Exception('Please specify the Windows domain for user in <domain>\<username> format')
            if not legacy.group(2):
                raise Exception('Please specify the Username of the user in <domain>\<username> format')
            if delegation is True:
                return HttpCredSSPAuth(legacy.group(1), legacy.group(2), password)
            else:
                return HttpNtlmAuth(legacy.group(1), legacy.group(2), password)

        #return HttpCredSSPAuth("SERVER2012", "Administrator", password)
        # attempt NTLM (local account, not domain) - if username is '' then we try anonymous NTLM auth
        # as if anyone will configure that - uf!
        return HttpNtlmAuth('', username, password)