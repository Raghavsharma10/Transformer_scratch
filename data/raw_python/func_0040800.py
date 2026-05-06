def diffuse(self, *args):
        """
        this is a dispatcher of diffuse implementation.
        Depending of the arguments used.
        """

        mode = diffusingModeEnum.unknown
        if (isinstance(args[0], str) and (len(args) == 3)):
            # reveived diffuse(str, any, any)
            mode = diffusingModeEnum.element

        elif (hasattr(args[0], "__len__") and (len(args) == 2)):
            # reveived diffuse(dict({str: any}), dict({str: any}))
            mode = diffusingModeEnum.elements

        else:
            raise TypeError(
                "Called diffuse method using bad argments, receive this" +
                " '{0}', but expected 'str, any, any' or" +
                " 'dict(str: any), dict(str: any)'."
                .format(args))

        self._diffuse(mode, *args)