def add_chunk(self,start,end,trig_start=0,trig_end=0):
    """
    Add an AnalysisChunk to the list associated with this ScienceSegment.
    @param start: GPS start time of chunk.
    @param end: GPS end time of chunk.
    @param trig_start: GPS start time for triggers from chunk
    """
    self.__chunks.append(AnalysisChunk(start,end,trig_start,trig_end))