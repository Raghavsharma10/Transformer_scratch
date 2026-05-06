def _cloneDefaultedParameter(self, original, default):
        """
        Make a copy of the parameter C{original}, supplying C{default} as the
        default value.

        @type original: L{Parameter} or L{ChoiceParameter}
        @param original: A liveform parameter.

        @param default: An alternate default value for the parameter.

        @rtype: L{Parameter} or L{ChoiceParameter}
        @return: A new parameter.
        """
        if isinstance(original, ChoiceParameter):
            default = [Option(o.description, o.value, o.value in default)
                        for o in original.choices]
        return original.clone(default)