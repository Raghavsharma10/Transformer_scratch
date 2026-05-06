def show_warning_messages(self, title=_(u"Incorrect Operation"), box_type='warning'):
        """
        It shows incorrect operations or successful operation messages.

        Args:
            title (string): title of message box
            box_type (string): type of message box (warning, info)
        """
        msg = self.current.task_data['msg']
        self.current.output['msgbox'] = {'type': box_type, "title": title, "msg": msg}
        del self.current.task_data['msg']