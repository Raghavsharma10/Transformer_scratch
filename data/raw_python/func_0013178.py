def bgp_time_conversion(bgp_uptime):
        """
        Convert string time to seconds.

        Examples
        00:14:23
        00:13:40
        00:00:21
        00:00:13
        00:00:49
        1d11h
        1d17h
        1w0d
        8w5d
        1y28w
        never
        """
        bgp_uptime = bgp_uptime.strip()
        uptime_letters = set(['w', 'h', 'd'])

        if 'never' in bgp_uptime:
            return -1
        elif ':' in bgp_uptime:
            times = bgp_uptime.split(":")
            times = [int(x) for x in times]
            hours, minutes, seconds = times
            return (hours * 3600) + (minutes * 60) + seconds
        # Check if any letters 'w', 'h', 'd' are in the time string
        elif uptime_letters & set(bgp_uptime):
            form1 = r'(\d+)d(\d+)h'  # 1d17h
            form2 = r'(\d+)w(\d+)d'  # 8w5d
            form3 = r'(\d+)y(\d+)w'  # 1y28w
            match = re.search(form1, bgp_uptime)
            if match:
                days = int(match.group(1))
                hours = int(match.group(2))
                return (days * DAY_SECONDS) + (hours * 3600)
            match = re.search(form2, bgp_uptime)
            if match:
                weeks = int(match.group(1))
                days = int(match.group(2))
                return (weeks * WEEK_SECONDS) + (days * DAY_SECONDS)
            match = re.search(form3, bgp_uptime)
            if match:
                years = int(match.group(1))
                weeks = int(match.group(2))
                return (years * YEAR_SECONDS) + (weeks * WEEK_SECONDS)
        raise ValueError("Unexpected value for BGP uptime string: {}".format(bgp_uptime))