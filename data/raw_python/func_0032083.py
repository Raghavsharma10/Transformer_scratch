def from_dict(cls, d):
        """
        Validates a dict instance and transforms it in a
        :py:class:`gemstone.core.structs.JsonRpcRequest`
        instance

        :param d: The dict instance
        :return: A :py:class:`gemstone.core.structs.JsonRpcRequest`
                 if everything goes well, or None if the validation fails
        """
        for key in ("method", "jsonrpc"):
            if key not in d:
                raise JsonRpcInvalidRequestError()

        # check jsonrpc version
        jsonrpc = d.get("jsonrpc", None)
        if jsonrpc != "2.0":
            raise JsonRpcInvalidRequestError()

        # check method
        method = d.get("method", None)
        if not method:
            raise JsonRpcInvalidRequestError()
        if not isinstance(method, str):
            raise JsonRpcInvalidRequestError()

        # params
        params = d.get("params", {})
        if not isinstance(params, (list, dict)):
            raise JsonRpcInvalidRequestError()

        req_id = d.get("id", None)
        if not isinstance(req_id, (int, str)) and req_id is not None:
            raise JsonRpcInvalidRequestError()

        extras = {k: d[k] for k in d if k not in ("jsonrpc", "id", "method", "params")}

        instance = cls(
            id=req_id,
            method=method,
            params=params,
            extra=extras
        )
        return instance