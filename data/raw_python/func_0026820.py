def reCurveFromEntireInputContour(self, inputContour):
        """
        Match if entire input contour matches entire output contour,
        allowing for different start point.
        """
        if self.clockwise:
            inputFlat = inputContour.clockwiseFlat
        else:
            inputFlat = inputContour.counterClockwiseFlat
        outputFlat = []
        for segment in self.segments:
            # XXX this could be expensive
            assert segment.segmentType == "flat"
            outputFlat += segment.points
        # test lengths
        haveMatch = False
        if len(inputFlat) == len(outputFlat):
            if inputFlat == outputFlat:
                haveMatch = True
            else:
                inputStart = inputFlat[0]
                if inputStart in outputFlat:
                    # there should be only one occurance of the point
                    # but handle it just in case
                    if outputFlat.count(inputStart) > 1:
                        startIndexes = [index for index, point in enumerate(outputFlat) if point == inputStart]
                    else:
                        startIndexes = [outputFlat.index(inputStart)]
                    # slice and dice to test possible orders
                    for startIndex in startIndexes:
                        test = outputFlat[startIndex:] + outputFlat[:startIndex]
                        if inputFlat == test:
                            haveMatch = True
                            break
        if haveMatch:
            # clear out the flat points
            self.segments = []
            # replace with the appropriate points from the input
            if self.clockwise:
                inputSegments = inputContour.clockwiseSegments
            else:
                inputSegments = inputContour.counterClockwiseSegments
            for inputSegment in inputSegments:
                self.segments.append(
                    OutputSegment(
                        segmentType=inputSegment.segmentType,
                        points=[
                            OutputPoint(
                                coordinates=point.coordinates,
                                segmentType=point.segmentType,
                                smooth=point.smooth,
                                name=point.name,
                                kwargs=point.kwargs
                            )
                            for point in inputSegment.points
                        ],
                        final=True
                    )
                )
                inputSegment.used = True
            # reset the direction of the final contour
            self.clockwise = inputContour.clockwise
            return True
        return False