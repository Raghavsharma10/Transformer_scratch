def from_parameter(cls: Type[XHXParameterType], parameter: str) -> Optional[XHXParameterType]:
        """
        Return a XHXParameter instance from an index parameter

        :param parameter: Index parameter

        :return:
        """
        xhx = XHXParameter.re_xhx.match(parameter)
        if xhx:
            return cls(int(xhx.group(1)))

        return None