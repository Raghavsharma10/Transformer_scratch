def navdatapush(self):
        """
        Pushes the current :referenceframe: out to clients.

        :return:
        """

        try:
            self.fireEvent(referenceframe({
                'data': self.referenceframe, 'ages': self.referenceages
            }), "navdata")
            self.intervalcount += 1

            if self.intervalcount == self.passiveinterval and len(
                    self.referenceframe) > 0:
                self.fireEvent(broadcast('users', {
                    'component': 'hfos.navdata.sensors',
                    'action': 'update',
                    'data': {
                        'data': self.referenceframe,
                        'ages': self.referenceages
                    }
                }), "hfosweb")
                self.intervalcount = 0
                # self.log("Reference frame successfully pushed.",
                # lvl=verbose)
        except Exception as e:
            self.log("Could not push referenceframe: ", e, type(e),
                     lvl=critical)