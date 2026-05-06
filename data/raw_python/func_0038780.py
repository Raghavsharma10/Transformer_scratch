def set_objective(self, measured_metabolites):
        '''
        Updates objective function for given measured metabolites.

        :param dict measured_metabolites: dict in which keys are metabolite names 
            and values are float numbers represent fold changes in metabolites. 
        '''
        self.clean_objective()
        for k, v in measured_metabolites.items():

            m = self.model.metabolites.get_by_id(k)
            total_stoichiometry = m.total_stoichiometry(
                self.without_transports)

            for r in m.producers(self.without_transports):
                update_rate = v * r.metabolites[m] / total_stoichiometry
                r.objective_coefficient += update_rate