def from_inline(cls: Type[UnlockType], inline: str) -> UnlockType:
        """
        Return an Unlock instance from inline string format

        :param inline: Inline string format

        :return:
        """
        data = Unlock.re_inline.match(inline)
        if data is None:
            raise MalformedDocumentError("Inline input")
        index = int(data.group(1))
        parameters_str = data.group(2).split(' ')
        parameters = []
        for p in parameters_str:
            param = UnlockParameter.from_parameter(p)
            if param:
                parameters.append(param)
        return cls(index, parameters)