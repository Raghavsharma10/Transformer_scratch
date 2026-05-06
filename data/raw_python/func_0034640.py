def _compute_response(response_key, server_challenge, client_challenge):
        """
        ComputeResponse() has been refactored slightly to reduce its complexity and improve
        readability, the 'if' clause which switches between LMv2 and NTLMv2 computation has been
        removed. Users should not call this method directly, they should rely on get_lmv2_response
        and get_ntlmv2_response depending on the negotiated flags.

        [MS-NLMP] v20140502 NT LAN Manager (NTLM) Authentication Protocol
        3.3.2 NTLM v2 Authentication
        """
        hmac_context = hmac.HMAC(response_key, hashes.MD5(), backend=default_backend())
        hmac_context.update(server_challenge)
        hmac_context.update(client_challenge)
        return hmac_context.finalize()