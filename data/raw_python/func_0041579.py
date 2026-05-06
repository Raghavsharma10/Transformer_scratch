def _get_ldap_msg(e):
        """Extract LDAP exception message"""
        msg = e
        if hasattr(e, 'message'):
            msg = e.message
            if 'desc' in e.message:
                msg = e.message['desc']
            elif hasattr(e, 'args'):
                msg = e.args[0]['desc']
        return msg