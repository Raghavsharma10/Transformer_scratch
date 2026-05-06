def link(self, next_stage):
        """Link to the given downstream stage *next_stage*
        by adding its input tube to the list of this stage's output tubes.
        Return this stage."""
        if next_stage is self: raise ValueError('cannot link stage to itself')
        self._output_tubes.append(next_stage._input_tube)
        self._next_stages.append(next_stage)
        return self