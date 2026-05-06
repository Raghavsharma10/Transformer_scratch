def deserialize(cls, value):
        """
        Generates a Peer instance via a JSON string of the sort generated
        by `Peer.deserialize`.

        The `name` and `ip` keys are required to be present in the JSON map,
        if the `port` key is not present the default is used.
        """
        parsed = json.loads(value)

        if "name" not in parsed:
            raise ValueError("No peer name.")
        if "ip" not in parsed:
            raise ValueError("No peer IP.")
        if "port" not in parsed:
            parsed["port"] = DEFAULT_PEER_PORT

        return cls(parsed["name"], parsed["ip"], parsed["port"])