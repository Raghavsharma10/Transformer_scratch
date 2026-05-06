def get_courses(self):
        """Gets any courses associated with this activity.

        return: (osid.course.CourseList) - list of courses
        raise:  IllegalState - ``is_course_based_activity()`` is
                ``false``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_assets_template
        if not bool(self._my_map['courseIds']):
            raise errors.IllegalState('no courseIds')
        mgr = self._get_provider_manager('COURSE')
        if not mgr.supports_course_lookup():
            raise errors.OperationFailed('Course does not support Course lookup')

        # What about the Proxy?
        lookup_session = mgr.get_course_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_no_catalog_view()
        return lookup_session.get_courses_by_ids(self.get_course_ids())