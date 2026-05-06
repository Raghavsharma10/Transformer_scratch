def preview_email_marketing_campaign(self, email_marketing_campaign):
        """Returns HTML and text previews of an EmailMarketingCampaign.
        """
        url = self.api.join('/'.join([
            self.EMAIL_MARKETING_CAMPAIGN_URL,
            str(email_marketing_campaign.constant_contact_id),
            'preview']))
        response = url.get()
        self.handle_response_status(response)
        return (response.json()['preview_email_content'],
                response.json()['preview_text_content'])