def _time_to_expiry(expires):
        """
        Determines the seconds until a HTTP header "Expires" timestamp
        :param expires: HTTP response "Expires" header
        :return: seconds until "Expires" time
        """
        try:
            expires_dt = datetime.strptime(str(expires), '%a, %d %b %Y %H:%M:%S %Z')
            delta = expires_dt - datetime.utcnow()
            return delta.seconds
        except ValueError:
            return 0