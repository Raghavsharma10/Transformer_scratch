def widget_status(self):
        """This method will return the status of all of the widgets in the
        widget list"""
        widget_status_list = []
        for i in self.widgetlist:
            widget_status_list += [[i.name, i.status]]
        return widget_status_list