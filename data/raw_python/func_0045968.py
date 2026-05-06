def match_any_learning_objective(self, match):
        """Matches an item with any objective.

        arg:    match (boolean): ``true`` to match items with any
                learning objective, ``false`` to match items with no
                learning objectives
        *compliance: mandatory -- This method must be implemented.*

        """
        match_key = 'learningObjectiveIds'
        param = '$exists'
        if match:
            flag = 'true'
        else:
            flag = 'false'
        if match_key in self._my_osid_query._query_terms:
            self._my_osid_query._query_terms[match_key][param] = flag
        else:
            self._my_osid_query._query_terms[match_key] = {param: flag}
        self._my_osid_query._query_terms[match_key]['$nin'] = [[], ['']]