def update(self, vrfs):
        """
        Method to update vrf's

        :param vrfs: List containing vrf's desired to updated
        :return: None
        """

        data = {'vrfs': vrfs}
        vrfs_ids = [str(vrf.get('id')) for vrf in vrfs]

        return super(ApiVrf, self).put('api/v3/vrf/%s/' %
                                       ';'.join(vrfs_ids), data)