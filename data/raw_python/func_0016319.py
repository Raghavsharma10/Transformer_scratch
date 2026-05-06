def create(self, Name, Subject, HtmlBody=None, TextBody=None, Alias=None):
        """
        Creates a template.

        :param Name: Name of template
        :param Subject: The content to use for the Subject when this template is used to send email.
        :param HtmlBody: The content to use for the HtmlBody when this template is used to send email.
        :param TextBody: The content to use for the HtmlBody when this template is used to send email.
        :return:
        """
        assert TextBody or HtmlBody, "Provide either email TextBody or HtmlBody or both"
        data = {"Name": Name, "Subject": Subject, "HtmlBody": HtmlBody, "TextBody": TextBody, "Alias": Alias}
        return self._init_instance(self.call("POST", "/templates", data=data))