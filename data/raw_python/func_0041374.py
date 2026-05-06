def from_parameter(cls: Type[SIGParameterType], parameter: str) -> Optional[SIGParameterType]:
        """
        Return a SIGParameter instance from an index parameter

        :param parameter: Index parameter

        :return:
        """
        sig = SIGParameter.re_sig.match(parameter)
        if sig:
            return cls(int(sig.group(1)))

        return None