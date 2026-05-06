def get_ntlmv1_response(password, challenge):
        """
        Generate the Unicode MD4 hash for the password associated with these credentials.
        """
        ntlm_hash = PasswordAuthentication.ntowfv1(password.encode('utf-16le'))
        response  = PasswordAuthentication._encrypt_des_block(ntlm_hash[:7], challenge)
        response += PasswordAuthentication._encrypt_des_block(ntlm_hash[7:14], challenge)
        response += PasswordAuthentication._encrypt_des_block(ntlm_hash[14:], challenge)

        # The NTLMv1 session key is simply the MD4 hash of the ntlm hash
        session_hash = hashlib.new('md4')
        session_hash.update(ntlm_hash)
        return response, session_hash.digest()