def from_str(cls: Type[BlockUIDType], blockid: str) -> BlockUIDType:
        """
        :param blockid: The block id
        """
        data = BlockUID.re_block_uid.match(blockid)
        if data is None:
            raise MalformedDocumentError("BlockUID")
        try:
            number = int(data.group(1))
        except AttributeError:
            raise MalformedDocumentError("BlockUID")

        try:
            sha_hash = data.group(2)
        except AttributeError:
            raise MalformedDocumentError("BlockHash")

        return cls(number, sha_hash)