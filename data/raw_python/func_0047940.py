def get_resource_relationships_for_source_resource_on_date(self, source_resource_id, from_, to):
        """Pass through to provider ResourceRelationshipLookupSession.get_resource_relationships_for_source_resource_on_date"""
        # Implemented from azosid template for -
        # osid.relationship.RelationshipLookupSession.get_relationships_for_source_on_date_template
        if self._can('lookup'):
            return self._provider_session.get_resource_relationships_for_source_resource_on_date(source_resource_id, from_, to)
        self._check_lookup_conditions()  # raises PermissionDenied
        query = self._query_session.get_resource_relationship_query()
        query.match_source_id(source_resource_id, match=True)
        query.match_date(from_, to, match=True)
        return self._try_harder(query)