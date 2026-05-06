def get_disease_pathways(self, disease_id=None, disease_name=None, pathway_id=None, pathway_name=None,
                             disease_definition=None, limit=None, as_df=False):
        """Get disease pathway link
        
        :param bool as_df: if set to True result returns as `pandas.DataFrame`
        :param disease_id: 
        :param disease_name: 
        :param pathway_id: 
        :param pathway_name:
        :param disease_definition:
        :param int limit: maximum number of results
        :return: list of :class:`pyctd.manager.database.models.DiseasePathway` objects

        .. seealso::
            
            :class:`pyctd.manager.models.DiseasePathway`

            which is linked to:
            :class:`pyctd.manager.models.Disease`
            :class:`pyctd.manager.models.Pathway`
        """
        q = self.session.query(models.DiseasePathway)

        q = self._join_disease(query=q, disease_id=disease_id, disease_name=disease_name,
                               disease_definition=disease_definition)

        q = self._join_pathway(query=q, pathway_id=pathway_id, pathway_name=pathway_name)

        return self._limit_and_df(q, limit, as_df)