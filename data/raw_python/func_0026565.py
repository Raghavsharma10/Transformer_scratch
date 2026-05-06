def activityrequest(self, event):
        """ActivityMonitor event handler for incoming events

        :param event with incoming ActivityMonitor message
        """

        # self.log("Event: '%s'" % event.__dict__)

        try:
            action = event.action
            data = event.data
            self.log("Activityrequest: ", action, data)

        except Exception as e:
            self.log("Error: '%s' %s" % (e, type(e)), lvl=error)