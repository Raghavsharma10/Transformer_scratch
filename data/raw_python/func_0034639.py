def ntowfv2(domain, user, password):
        """
        NTOWFv2() Implementation
        [MS-NLMP] v20140502 NT LAN Manager (NTLM) Authentication Protocol
        3.3.2 NTLM v2 Authentication
        :param domain: The windows domain name
        :param user: The windows username
        :param password: The users password
        :return: Hash Data
        """
        md4 = hashlib.new('md4')
        md4.update(password)
        hmac_context = hmac.HMAC(md4.digest(), hashes.MD5(), backend=default_backend())
        hmac_context.update(user.upper().encode('utf-16le'))
        hmac_context.update(domain.encode('utf-16le'))
        return hmac_context.finalize()