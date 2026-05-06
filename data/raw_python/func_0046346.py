def get_learning_objectives(self):
        """ This method also mirrors that in the Item."""
        # This is pretty much identicial to the method in assessment.Item!
        mgr = self._get_provider_manager('LEARNING')
        lookup_session = mgr.get_objective_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_objective_bank_view()
        return lookup_session.get_objectives_by_ids(self.get_learning_objective_ids())