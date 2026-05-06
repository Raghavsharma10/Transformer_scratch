def set_courses(self, course_ids):
        """Sets the courses.

        arg:    course_ids (osid.id.Id[]): the course ``Ids``
        raise:  InvalidArgument - ``course_ids`` is invalid
        raise:  NullArgument - ``course_ids`` is ``null``
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.ActivityForm.set_assets_template
        if not isinstance(course_ids, list):
            raise errors.InvalidArgument()
        if self.get_courses_metadata().is_read_only():
            raise errors.NoAccess()
        idstr_list = []
        for object_id in course_ids:
            if not self._is_valid_id(object_id):
                raise errors.InvalidArgument()
            idstr_list.append(str(object_id))
        self._my_map['courseIds'] = idstr_list