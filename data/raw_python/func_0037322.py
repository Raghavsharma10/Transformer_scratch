def get_disease(self, disease_name=None, disease_id=None, definition=None, parent_ids=None, tree_numbers=None,
                    parent_tree_numbers=None, slim_mapping=None, synonym=None, alt_disease_id=None, limit=None,
                    as_df=False):
        """
        Get diseases

        :param bool as_df: if set to True result returns as `pandas.DataFrame`
        :param int limit: maximum number of results
        :param str disease_name: disease name
        :param str disease_id: disease identifier
        :param str definition: definition of disease
        :param str parent_ids: parent identifiers, delimiter |
        :param str tree_numbers: tree numbers, delimiter |
        :param str parent_tree_numbers: parent tree numbers, delimiter
        :param str slim_mapping:  term derived from the MeSH tree structure for the “Diseases” [C] branch, \
        that classifies MEDIC diseases into high-level categories
        :param str synonym: disease synonyms
        :param str alt_disease_id: alternative disease identifiers
        :return: list of :class:`pyctd.manager.models.Disease` object


        .. seealso::

            :class:`pyctd.manager.models.Disease`

        .. todo::
            normalize parent_ids, tree_numbers and parent_tree_numbers in :class:`pyctd.manager.models.Disease`
        """
        q = self.session.query(models.Disease)

        if disease_name:
            q = q.filter(models.Disease.disease_name.like(disease_name))

        if disease_id:
            q = q.filter(models.Disease.disease_id == disease_id)

        if definition:
            q = q.filter(models.Disease.definition.like(definition))

        if parent_ids:
            q = q.filter(models.Disease.parent_ids.like(parent_ids))

        if tree_numbers:
            q = q.filter(models.Disease.tree_numbers.like(tree_numbers))

        if parent_tree_numbers:
            q = q.filter(models.Disease.parent_tree_numbers.like(parent_tree_numbers))

        if slim_mapping:
            q = q.join(models.DiseaseSlimmapping).filter(models.DiseaseSlimmapping.slim_mapping.like(slim_mapping))

        if synonym:
            q = q.join(models.DiseaseSynonym).filter(models.DiseaseSynonym.synonym.like(synonym))

        if alt_disease_id:
            q = q.join(models.DiseaseAltdiseaseid).filter(models.DiseaseAltdiseaseid.alt_disease_id == alt_disease_id)

        return self._limit_and_df(q, limit, as_df)