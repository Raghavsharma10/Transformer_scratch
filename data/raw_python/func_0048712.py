def get_agent_metadata(self):
        """Gets the metadata for the agent.

        return: (osid.Metadata) - metadata for the agent
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['agent'])
        metadata.update({'existing_id_values': self._my_map['agentId']})
        return Metadata(**metadata)