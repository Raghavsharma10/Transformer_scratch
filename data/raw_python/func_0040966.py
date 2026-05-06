def get_template_options(self, instance=None, test_message=None, **kwargs):
        """Returns a dictionary of message template options.

        Extend using `extra_template_options`.
        """
        protocol_name = django_apps.get_app_config("edc_protocol").protocol_name
        test_message = test_message or self.test_message
        template_options = dict(
            name=self.name,
            protocol_name=protocol_name,
            display_name=self.display_name,
            email_from=self.email_from,
            test_subject_line=(
                self.email_test_subject_line if test_message else ""
            ).strip(),
            test_body_line=self.email_test_body_line if test_message else "",
            test_line=self.sms_test_line if test_message else "",
            message_datetime=get_utcnow(),
            message_reference="",
        )
        if "subject_identifier" not in template_options:
            try:
                template_options.update(subject_identifier=instance.subject_identifier)
            except AttributeError:
                pass
        if "site_name" not in template_options:
            try:
                template_options.update(site_name=instance.site.name.title())
            except AttributeError:
                pass
        return template_options