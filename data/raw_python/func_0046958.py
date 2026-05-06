def clear_cognitive_process(self):
        """Clears the cognitive process.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.clear_avatar_template
        if (self.get_cognitive_process_metadata().is_read_only() or
                self.get_cognitive_process_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['cognitiveProcessId'] = self._cognitive_process_default