def populate_data(self, data):
        """Parse the outgoing schema"""
        sch = MockItemSchema()
        return Result(**{
            "callname": self.context.get("callname"),
            "result": sch.dump(data),
        })