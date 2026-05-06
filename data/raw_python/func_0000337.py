def tplds(self):
        """
        :return: dictionary {id: object} of all current tplds.
        :rtype: dict of (int, xenamanager.xena_port.XenaTpld)
        """

        # As TPLDs are dynamic we must re-read them each time from the port.
        self.parent.del_objects_by_type('tpld')
        for tpld in self.get_attribute('pr_tplds').split():
            XenaTpld(parent=self, index='{}/{}'.format(self.index, tpld))
        return {t.id: t for t in self.get_objects_by_type('tpld')}