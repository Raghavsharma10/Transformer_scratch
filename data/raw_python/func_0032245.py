def getParameters(self, phoneNumber):
        """
        Return a C{list} of two liveform parameters, one for editing
        C{phoneNumber}'s I{number} attribute, and one for editing its I{label}
        attribute.

        @type phoneNumber: L{PhoneNumber} or C{NoneType}
        @param phoneNumber: If not C{None}, an existing contact item from
        which to get the parameter's default values.

        @rtype: C{list}
        """
        defaultNumber = u''
        defaultLabel = PhoneNumber.LABELS.HOME
        if phoneNumber is not None:
            defaultNumber = phoneNumber.number
            defaultLabel = phoneNumber.label
        labelChoiceParameter = liveform.ChoiceParameter(
            'label',
            [liveform.Option(label, label, label == defaultLabel)
                for label in PhoneNumber.LABELS.ALL_LABELS],
            'Number Type')
        return [
            labelChoiceParameter,
            liveform.Parameter(
                'number',
                liveform.TEXT_INPUT,
                unicode,
                'Phone Number',
                default=defaultNumber)]