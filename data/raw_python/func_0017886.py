def _get_auth_packet(self, username, password, client):
        """
        Get the pyrad authentication packet for the username/password and the
        given pyrad client.
        """
        pkt = client.CreateAuthPacket(code=AccessRequest,
                                      User_Name=username)
        pkt["User-Password"] = pkt.PwCrypt(password)
        pkt["NAS-Identifier"] = 'django-radius'
        for key, val in list(getattr(settings, 'RADIUS_ATTRIBUTES', {}).items()):
            pkt[key] = val
        return pkt