def _list_patient_ids(self):
        """ Utility function to return a list of patient ids in the Cohort
        """
        results = []
        for patient in self:
            results.append(patient.id)
        return(results)