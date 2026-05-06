def update(self) -> None:
        """Reference the actual |Indexer.timeofyear| array of the
        |Indexer| object available in module |pub|.

        >>> from hydpy import pub
        >>> pub.timegrids = '27.02.2004', '3.03.2004', '1d'
        >>> from hydpy.core.parametertools import TOYParameter
        >>> toyparameter = TOYParameter(None)
        >>> toyparameter.update()
        >>> toyparameter
        toyparameter(57, 58, 59, 60, 61)
        """
        indexarray = hydpy.pub.indexer.timeofyear
        self.shape = indexarray.shape
        self.values = indexarray