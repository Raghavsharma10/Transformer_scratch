def get_knowledge_category_metadata(self):
        """Gets the metadata for a knowledge category.

        return: (osid.Metadata) - metadata for the knowledge category
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.get_group_metadata_template
        metadata = dict(self._mdata['knowledge_category'])
        metadata.update({'existing_id_values': self._my_map['knowledgeCategoryId']})
        return Metadata(**metadata)