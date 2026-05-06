def call(self, inpt):
        """
        Returns if the condition applies to the ``inpt``.

        If the class ``inpt`` is an instance of is not the same class as the
        condition's own ``argument``, then ``False`` is returned.  This also
        applies to the ``NONE`` input.

        Otherwise, ``argument`` is called, with ``inpt`` as the instance and
        the value is compared to the ``operator`` and the Value is returned.  If
        the condition is ``negative``, then then ``not`` the value is returned.

        Keyword Arguments:
        inpt -- An instance of the ``Input`` class.
        """
        if inpt is Manager.NONE_INPUT:
            return False

        # Call (construct) the argument with the input object
        argument_instance = self.argument(inpt)

        if not argument_instance.applies:
            return False

        application = self.__apply(argument_instance, inpt)

        if self.negative:
            application = not application

        return application