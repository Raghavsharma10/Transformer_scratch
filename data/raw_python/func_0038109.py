def get_context(self):
        """Add mails to the context

        """
        context = super(MailListView, self).get_context()
        mail_list = registered_mails_names()

        context['mail_map'] = mail_list
        return context