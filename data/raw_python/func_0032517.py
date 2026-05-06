def options(self, request, tag):
        """
        Render each of the options of the wrapped L{ChoiceParameter} instance.
        """
        option = tag.patternGenerator('option')
        return tag[[
                OptionView(index, o, option())
                for (index, o)
                in enumerate(self.parameter.choices)]]