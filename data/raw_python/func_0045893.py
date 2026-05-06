def get_id(self):
        """override get_id to generate our "magic" ids that encode choice order"""

        # Check first to make sure no one else has claimed authority on my object.
        # This will likely occur when an AssessmentSection returns a Question
        # During an AssessmentSession
        if self.my_osid_object._authority != MAGIC_AUTHORITY:
            return self.my_osid_object._item_id
            # raise AttributeError

        # If not, go ahead and build magic Id:
        choices = self.my_osid_object._my_map['choices']
        choice_ids = [c['id'] for c in choices]
        magic_identifier = quote('{0}?{1}'.format(self.my_osid_object._my_map['_id'],
                                                  json.dumps(choice_ids)))
        return Id(namespace='assessment.Item',
                  identifier=magic_identifier,
                  authority=MAGIC_AUTHORITY)