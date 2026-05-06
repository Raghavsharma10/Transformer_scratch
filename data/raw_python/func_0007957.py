def summary(self, campaign_id=None):
        """ Returns the campaign summary """
        resource_cls = CampaignSummary
        single_resource = False

        if not campaign_id:
            resource_cls = CampaignSummaries
            single_resource = True

        return super(API, self).get(
            resource_id=campaign_id,
            resource_action='summary',
            resource_cls=resource_cls,
            single_resource=single_resource)