def save_controls(self, parameterstep: 'timetools.PeriodConstrArg' = None,
                      simulationstep: 'timetools.PeriodConstrArg' = None,
                      auxfiler: 'Optional[auxfiletools.Auxfiler]' = None):
        """Save the control parameters of the |Model| object handled by
        each |Element| object and eventually the ones handled by the
        given |Auxfiler| object."""
        if auxfiler:
            auxfiler.save(parameterstep, simulationstep)
        for element in printtools.progressbar(self):
            element.model.parameters.save_controls(
                parameterstep=parameterstep,
                simulationstep=simulationstep,
                auxfiler=auxfiler)