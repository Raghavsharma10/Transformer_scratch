def delete_email_marketing_campaign(self, email_marketing_campaign):
        """Deletes a Constant Contact email marketing campaign.
        """
        url = self.api.join('/'.join([
            self.EMAIL_MARKETING_CAMPAIGN_URL,
            str(email_marketing_campaign.constant_contact_id)]))
        response = url.delete()
        self.handle_response_status(response)
        return response