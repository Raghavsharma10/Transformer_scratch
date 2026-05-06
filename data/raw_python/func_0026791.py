def handle_webhook_event(self, environ, url, params):
        """
        Webhook handler - each handler for the webhook event
        takes an initial pattern argument for matching the URL
        requested. Here we match the URL to the pattern for each
        webhook handler, and bail out if it returns a response.
        """
        for handler in self.events["webhook"]:
            urlpattern = handler.event.args["urlpattern"]
            if not urlpattern or match(urlpattern, url):
                response = handler(self, environ, url, params)
                if response:
                    return response