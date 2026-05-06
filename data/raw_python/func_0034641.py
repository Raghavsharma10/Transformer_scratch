def get_lmv2_response(domain, username, password, server_challenge, client_challenge):
        """
        Computes an appropriate LMv2 response based on the supplied arguments
        The algorithm is based on jCIFS. The response is 24 bytes, with the 16 bytes of hash
        concatenated with the 8 byte client client_challenge
        """
        ntlmv2_hash = PasswordAuthentication.ntowfv2(domain, username, password.encode('utf-16le'))
        hmac_context = hmac.HMAC(ntlmv2_hash, hashes.MD5(), backend=default_backend())
        hmac_context.update(server_challenge)
        hmac_context.update(client_challenge)
        lmv2_hash = hmac_context.finalize()

        # The LMv2 master user session key is a HMAC MD5 of the NTLMv2 and LMv2 hash
        session_key = hmac.HMAC(ntlmv2_hash, hashes.MD5(), backend=default_backend())
        session_key.update(lmv2_hash)

        return lmv2_hash + client_challenge, session_key.finalize()