def Solve(self, stops=None, barriers=None, returnDirections=None,
              returnRoutes=None, returnStops=None, returnBarriers=None,
              outSR=None, ignoreInvalidLocations=None, outputLines=None,
              findBestSequence=None, preserveFirstStop=None,
              preserveLastStop=None, useTimeWindows=None, startTime=None, 
              accumulateAttributeNames=None, impedanceAttributeName=None,
              restrictionAttributeNames=None, restrictUTurns=None,
              useHierarchy=None, directionsLanguage=None,
              outputGeometryPrecision=None, directionsLengthUnits=None,
              directionsTimeAttributeName=None, attributeParameterValues=None,
              polylineBarriers=None, polygonBarriers=None):
        """The solve operation is performed on a network layer resource.

           At 9.3.1, the solve operation is supported only on the route layer.
           Or specifically, on a network layer whose layerType is
           esriNAServerRouteLayer.

           You can provide arguments to the solve route operation as query
           parameters defined in the parameters table below.
        """
        def ptlist_as_semilist(lst):
            if isinstance(lst, geometry.Point):
                lst = [lst]
            if isinstance(lst, (list, tuple)):
                return ";".join(','.join(str(x) for x in pt) for pt in lst)
            return lst
        if self.layerType != "esriNAServerRouteLayer":
            raise TypeError("Layer is of type %s; Solve is not available."
                            % self.layerType)
        return self._get_subfolder('solve/', NetworkSolveResult,
                       {'stops': ptlist_as_semilist(stops),
                        'barriers': ptlist_as_semilist(barriers),
                        'returnDirections': returnDirections,
                        'returnRoutes': returnRoutes,
                        'returnStops': returnStops,
                        'returnBarriers': returnBarriers,
                        'outSR': outSR,
                        'ignoreInvalidLocations': ignoreInvalidLocations,
                        'outputLines': outputLines,
                        'findBestSequence': findBestSequence,
                        'preserveFirstStop': preserveFirstStop,
                        'preserveLastStop': preserveLastStop,
                        'useTimeWindows': useTimeWindows,
                        'startTime': startTime,
                        'accumulateAttributeNames': accumulateAttributeNames,
                        'impedanceAttributeName': impedanceAttributeName,
                        'restrictionAttributeNames': restrictionAttributeNames,
                        'restrictUTurns': restrictUTurns,
                        'useHierarchy': useHierarchy,
                        'directionsLanguage': directionsLanguage,
                        'outputGeometryPrecision': outputGeometryPrecision,
                        'directionsLengthUnits': directionsLengthUnits,
                        'directionsTimeAttributeName':
                                                  directionsTimeAttributeName,
                        'attributeParameterValues': attributeParameterValues,
                        'polylineBarriers': polylineBarriers,
                        'polygonBarriers': polygonBarriers})