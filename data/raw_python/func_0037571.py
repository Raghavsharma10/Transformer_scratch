def save(self, *args, **kwargs):
        """Kicks off celery task to re-save associated special coverages to percolator

        :param args: inline arguments (optional)
        :param kwargs: keyword arguments
        :return: `bulbs.campaigns.Campaign`
        """
        campaign = super(Campaign, self).save(*args, **kwargs)
        save_campaign_special_coverage_percolator.delay(self.tunic_campaign_id)
        return campaign