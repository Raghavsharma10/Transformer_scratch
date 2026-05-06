def from_inline(cls: Type[OutputSourceType], inline: str) -> OutputSourceType:
        """
        Return OutputSource instance from inline string format

        :param inline: Inline string format
        :return:
        """
        data = OutputSource.re_inline.match(inline)
        if data is None:
            raise MalformedDocumentError("Inline output")
        amount = int(data.group(1))
        base = int(data.group(2))
        condition_text = data.group(3)

        return cls(amount, base, condition_text)