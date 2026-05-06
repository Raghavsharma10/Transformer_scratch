def deserialize(cls, value):
        """
        Creates a new Node instance via a JSON map string.

        Note that `port` and `ip` and are required keys for the JSON map,
        `peer` and `host` are optional.  If `peer` is not present, the new Node
        instance will use the current peer.  If `host` is not present, the
        hostname of the given `ip` is looked up.
        """
        if getattr(value, "decode", None):
            value = value.decode()

        logger.debug("Deserializing node data: '%s'", value)
        parsed = json.loads(value)

        if "port" not in parsed:
            raise ValueError("No port defined for node.")
        if "ip" not in parsed:
            raise ValueError("No IP address defined for node.")
        if "host" not in parsed:
            host, aliases, ip_list = socket.gethostbyaddr(parsed["ip"])
            parsed["host"] = socket.get_fqdn(host)
        if "peer" in parsed:
            peer = Peer.deserialize(parsed["peer"])
        else:
            peer = None

        return cls(
            parsed["host"], parsed["ip"], parsed["port"],
            peer=peer, metadata=parsed.get("metadata")
        )