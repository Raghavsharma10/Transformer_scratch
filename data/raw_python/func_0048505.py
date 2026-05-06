def _init_map(self, record_types=None, **kwargs):
        """Initialize form map"""
        osid_objects.OsidRelationshipForm._init_map(self, record_types=record_types)
        self._my_map['assignedVaultIds'] = [str(kwargs['vault_id'])]
        self._my_map['functionId'] = str(kwargs['function_id'])
        self._my_map['qualifierId'] = str(kwargs['qualifier_id'])
        self._my_map['agentId'] = None
        self._my_map['resourceId'] = None
        self._my_map['trustId'] = None
        self._my_map['implicit'] = None
        if 'agent_id' in kwargs:
            self._my_map['agentId'] = str(kwargs['agent_id'])
        if 'resource_id' in kwargs:
            self._my_map['resourceId'] = str(kwargs['resource_id'])