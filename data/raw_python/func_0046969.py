def get_courses_metadata(self):
        """Gets the metadata for the courses.

        return: (osid.Metadata) - metadata for the courses
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.ActivityForm.get_assets_metadata_template
        metadata = dict(self._mdata['courses'])
        metadata.update({'existing_courses_values': self._my_map['courseIds']})
        return Metadata(**metadata)