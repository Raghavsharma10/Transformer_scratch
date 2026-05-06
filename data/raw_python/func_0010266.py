def from_json(cls, json_data):
        """Build and return a new Group object from json data (used internally)"""
        # Example Data:
        # { "grpId": "11817", "grpName": "7603_Digi", "grpDescription": "7603_Digi root group",
        #   "grpPath": "\/7603_Digi\/", "grpParentId": "1"}
        return cls(
            group_id=json_data["grpId"],
            name=json_data["grpName"],
            description=json_data.get("grpDescription", ""),
            path=json_data["grpPath"],
            parent_id=json_data["grpParentId"],
        )