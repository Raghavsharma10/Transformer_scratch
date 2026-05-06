def complete(self, campaign_id):
        """ Complete an existing campaign (Stop processing events) """

        return super(API, self).get(
            resource_id=campaign_id, resource_action='complete')