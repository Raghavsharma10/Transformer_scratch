def update_subscriptions(self, event):
        """OM event handler for to be stored and client shared objectmodels
        :param event: OMRequest with uuid, schema and object data
        """

        # self.log("Event: '%s'" % event.__dict__)
        try:
            self._update_subscribers(event.schema, event.data)

        except Exception as e:
            self.log("Error during subscription update: ", type(e), e,
                     exc=True)