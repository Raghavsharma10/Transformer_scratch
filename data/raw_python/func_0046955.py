def clear_knowledge_category(self):
        """Clears the knowledge category.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.clear_avatar_template
        if (self.get_knowledge_category_metadata().is_read_only() or
                self.get_knowledge_category_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['knowledgeCategoryId'] = self._knowledge_category_default