def validate_proxies_config(cls, proxies):
        """
        Specific config validation method for the "proxies" portion of a
        config.

        Checks that each proxy defines a port and a list of `upstreams`,
        and that each upstream entry has a host and port defined.
        """
        for name, proxy in six.iteritems(proxies):
            if "port" not in proxy:
                raise ValueError("No port defined for proxy %s" % name)
            if "upstreams" not in proxy:
                raise ValueError(
                    "No upstreams defined for proxy %s" % name
                )
            for upstream in proxy["upstreams"]:
                if "host" not in upstream:
                    raise ValueError(
                        "No host defined for upstream in proxy %s" % name
                    )
                if "port" not in upstream:
                    raise ValueError(
                        "No port defined for upstream in proxy %s" % name
                    )