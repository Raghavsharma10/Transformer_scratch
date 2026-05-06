def form(self, request, tag):
        """
        Render the inputs for a form.

        @param tag: A tag with:
            - I{form} and I{description} slots
            - I{liveform} and I{subform} patterns, to fill the I{form} slot
                - An I{inputs} slot, to fill with parameter views
            - L{IParameterView.patternName}I{-input-container} patterns for
              each parameter type in C{self.parameters}
        """
        patterns = PatternDictionary(self.docFactory)
        inputs = []

        for parameter in self.parameters:
            view = parameter.viewFactory(parameter, None)
            if view is not None:
                view.setDefaultTemplate(
                    tag.onePattern(view.patternName + '-input-container'))
                setFragmentParent = getattr(view, 'setFragmentParent', None)
                if setFragmentParent is not None:
                    setFragmentParent(self)
                inputs.append(view)
            else:
                inputs.append(_legacySpecialCases(self, patterns, parameter))

        if self.subFormName is None:
            pattern = tag.onePattern('liveform')
        else:
            pattern = tag.onePattern('subform')

        return dictFillSlots(
            tag,
            dict(form=pattern.fillSlots('inputs', inputs),
                 description=self._getDescription()))