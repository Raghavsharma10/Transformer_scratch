def set_poolmember_state(self, id_pools, pools):
        """
        Enable/Disable pool member by list
        """

        data = dict()

        uri = "api/v3/pool/real/%s/member/status/" % ';'.join(id_pools)

        data["server_pools"] = pools

        return self.put(uri, data=data)