def track(self, tracking_number):
        "Track a UPS package by number. Returns just a delivery date."
        resp = self.send_request(tracking_number)
        return self.parse_response(resp)