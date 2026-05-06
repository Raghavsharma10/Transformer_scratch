def get_proficiencies_for_resource_on_date(self, resource_id, from_, to):
        """Pass through to provider ProficiencyLookupSession.get_proficiencies_for_resource_on_date"""
        # Implemented from azosid template for -
        # osid.relationship.RelationshipLookupSession.get_relationships_for_source_on_date_template
        if self._can('lookup'):
            return self._provider_session.get_proficiencies_for_resource_on_date(resource_id, from_, to)
        self._check_lookup_conditions()  # raises PermissionDenied
        query = self._query_session.get_proficiency_query()
        query.match_source_id(resource_id, match=True)
        query.match_date(from_, to, match=True)
        return self._try_harder(query)