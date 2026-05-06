def update_email_marketing_campaign(self, email_marketing_campaign,
                                        name, email_content, from_email,
                                        from_name, reply_to_email, subject,
                                        text_content, address,
                                        is_view_as_webpage_enabled=False,
                                        view_as_web_page_link_text='',
                                        view_as_web_page_text='',
                                        is_permission_reminder_enabled=False,
                                        permission_reminder_text=''):
        """Update a Constant Contact email marketing campaign.
        Returns the updated EmailMarketingCampaign object.
        """
        url = self.api.join(
            '/'.join([self.EMAIL_MARKETING_CAMPAIGN_URL,
                      str(email_marketing_campaign.constant_contact_id)]))

        inlined_email_content = self.inline_css(email_content)
        minified_email_content = html_minify(inlined_email_content)
        worked_around_email_content = work_around(minified_email_content)

        data = {
            'name': name,
            'subject': subject,
            'from_name': from_name,
            'from_email': from_email,
            'reply_to_email': reply_to_email,
            'email_content': worked_around_email_content,
            'email_content_format': 'HTML',
            'text_content': text_content,
            'message_footer': {
                'organization_name': address['organization_name'],
                'address_line_1': address['address_line_1'],
                'address_line_2': address['address_line_2'],
                'address_line_3': address['address_line_3'],
                'city': address['city'],
                'state': address['state'],
                'international_state': address['international_state'],
                'postal_code': address['postal_code'],
                'country': address['country']
            },
            'is_view_as_webpage_enabled': is_view_as_webpage_enabled,
            'view_as_web_page_link_text': view_as_web_page_link_text,
            'view_as_web_page_text': view_as_web_page_text,
            'is_permission_reminder_enabled': is_permission_reminder_enabled,
            'permission_reminder_text': permission_reminder_text
        }

        response = url.put(data=json.dumps(data),
                           headers={'content-type': 'application/json'})

        self.handle_response_status(response)

        email_marketing_campaign.data = response.json()
        email_marketing_campaign.save()

        return email_marketing_campaign