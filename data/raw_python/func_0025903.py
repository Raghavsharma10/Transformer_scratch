def get_dc_owner(raises, mask_if_self):
    """ Convenience function to return owner of /dev/console.
    If raises is True, this raises an exception on any error.
    If not, it returns any error string as the owner name.
    If owner is self, and if mask_if_self, returns "<self>"."""
    try:
        from pwd import getpwuid
        owner_uid = os.stat('/dev/console').st_uid
        self_uid  = os.getuid()
        if mask_if_self and owner_uid == self_uid:
            return "<self>"
        owner_name = getpwuid(owner_uid).pw_name
        return owner_name
    except Exception as e:
        if raises:
            raise e
        else:
            return str(e)