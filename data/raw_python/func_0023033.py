def del_widget(self, ref):
        """ Delete/Remove A Widget """
        self.server.request("widget_del %s %s" % (self.name, ref))
        del(self.widgets[ref])