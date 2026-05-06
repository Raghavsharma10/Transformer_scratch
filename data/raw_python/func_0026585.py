def _handleAnonymousEvents(self, component, action, data, client):
        """Handler for anonymous (public) events"""
        try:
            event = self.anonymous_events[component][action]['event']

            self.log("Firing anonymous event: ", component, action,
                     str(data)[:20], lvl=network)
            # self.log("", (user, action, data, client), lvl=critical)
            self.fireEvent(event(action, data, client))
        except Exception as e:
            self.log("Critical error during anonymous event handling:",
                     component, action, e,
                     type(e), lvl=critical, exc=True)