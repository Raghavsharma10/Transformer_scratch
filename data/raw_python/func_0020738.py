def matches_ip(cls, ip_str, read_preference=None):
        """
        Return True if provided IP exists in the blacklist and doesn't exist
        in the whitelist. Otherwise, return False.
        """
        qs = cls.qs_for_ip(ip_str).only('whitelist')
        if read_preference:
            qs = qs.read_preference(read_preference)

        # Return True if any docs match the IP and none of them represent
        # a whitelist
        return bool(qs) and not any(obj.whitelist for obj in qs)