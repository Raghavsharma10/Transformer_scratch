def get_cognitive_process_metadata(self):
        """Gets the metadata for a cognitive process.

        return: (osid.Metadata) - metadata for the cognitive process
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['cognitive_process'])
        metadata.update({'existing_id_values': self._my_map['cognitiveProcessId']})
        return Metadata(**metadata)