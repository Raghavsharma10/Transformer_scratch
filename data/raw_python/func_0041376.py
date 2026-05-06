def from_parameter(cls: Type[UnlockParameterType], parameter: str) -> Optional[Union[SIGParameter, XHXParameter]]:
        """
        Return UnlockParameter instance from parameter string

        :param parameter: Parameter string
        :return:
        """

        sig_param = SIGParameter.from_parameter(parameter)
        if sig_param:
            return sig_param
        else:
            xhx_param = XHXParameter.from_parameter(parameter)
            if xhx_param:
                return xhx_param

        return None