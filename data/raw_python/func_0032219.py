def _parametersToDefaults(self, parameters):
        """
        Extract the defaults from C{parameters}, constructing a dictionary
        mapping parameter names to default values, suitable for passing to
        L{ListChangeParameter}.

        @type parameters: C{list} of L{liveform.Parameter} or
        L{liveform.ChoiceParameter}.

        @rtype: C{dict}
        """
        defaults = {}
        for p in parameters:
            if isinstance(p, liveform.ChoiceParameter):
                selected = []
                for choice in p.choices:
                    if choice.selected:
                        selected.append(choice.value)
                defaults[p.name] = selected
            else:
                defaults[p.name] = p.default
        return defaults