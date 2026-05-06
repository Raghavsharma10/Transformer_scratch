def clear_learning_objectives(self):
        """Clears the learning objectives.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.ActivityForm.clear_assets_template
        if (self.get_learning_objectives_metadata().is_read_only() or
                self.get_learning_objectives_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['learningObjectiveIds'] = self._learning_objectives_default