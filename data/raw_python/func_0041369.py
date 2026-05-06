def from_inline(cls: Type[InputSourceType], tx_version: int, inline: str) -> InputSourceType:
        """
        Return Transaction instance from inline string format

        :param tx_version: Version number of the document
        :param inline: Inline string format
        :return:
        """
        if tx_version == 2:
            data = InputSource.re_inline.match(inline)
            if data is None:
                raise MalformedDocumentError("Inline input")
            source_offset = 0
            amount = 0
            base = 0
        else:
            data = InputSource.re_inline_v3.match(inline)
            if data is None:
                raise MalformedDocumentError("Inline input")
            source_offset = 2
            amount = int(data.group(1))
            base = int(data.group(2))
        if data.group(1 + source_offset):
            source = data.group(1 + source_offset)
            origin_id = data.group(2 + source_offset)
            index = int(data.group(3 + source_offset))
        else:
            source = data.group(4 + source_offset)
            origin_id = data.group(5 + source_offset)
            index = int(data.group(6 + source_offset))

        return cls(amount, base, source, origin_id, index)