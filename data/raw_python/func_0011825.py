def make_request(self, data):
        """Parse the outgoing schema"""
        sch = MockItemSchema()
        return Request(**{
            "callname": self.context.get("callname"),
            "payload": sch.dump(data),
        })