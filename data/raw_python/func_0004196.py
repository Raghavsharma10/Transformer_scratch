def get_poolmember_state(self, id_pools, checkstatus=0):
        """
        Enable/Disable pool member by list
        """

        uri = "api/v3/pool/real/%s/member/status/?checkstatus=%s" % (';'.join(id_pools), checkstatus)

        return self.get(uri)