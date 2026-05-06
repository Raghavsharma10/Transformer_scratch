def _coerceAll(self, inputs):
        """
        XXX
        """
        def associate(result, obj):
            return (obj, result)

        coerceDeferreds = []
        for obj, dataSet in inputs:
            oneCoerce = self._coerceSingleRepetition(dataSet)
            oneCoerce.addCallback(associate, obj)
            coerceDeferreds.append(oneCoerce)
        return gatherResults(coerceDeferreds)