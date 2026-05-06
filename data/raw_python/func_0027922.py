def get_student_item_dict(self, anonymous_user_id=None):
        """Create a student_item_dict from our surrounding context.

        See also: submissions.api for details.

        Args:
            anonymous_user_id(str): A unique anonymous_user_id for (user, course) pair.
        Returns:
            (dict): The student item associated with this XBlock instance. This
                includes the student id, item id, and course id.
        """

        item_id = self._serialize_opaque_key(self.scope_ids.usage_id)

        # This is not the real way course_ids should work, but this is a
        # temporary expediency for LMS integration
        if hasattr(self, "xmodule_runtime"):
            course_id = self.get_course_id()  # pylint:disable=E1101

            if anonymous_user_id:
                student_id = anonymous_user_id
            else:
                student_id = self.xmodule_runtime.anonymous_student_id  # pylint:disable=E1101
        else:
            course_id = "edX/Enchantment_101/April_1"
            if self.scope_ids.user_id is None:
                student_id = ''
            else:
                student_id = unicode(self.scope_ids.user_id)

        student_item_dict = dict(
            student_id=student_id,
            item_id=item_id,
            course_id=course_id,
            item_type='ubcpi'
        )
        return student_item_dict