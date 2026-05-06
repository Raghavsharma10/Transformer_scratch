def parse(self, ioc_obj):
        """
        parses an ioc to populate self.iocs and self.ioc_name

        :param ioc_obj:
        :return:
        """
        if ioc_obj is None:
            return
        iocid = ioc_obj.iocid
        try:
            sd = ioc_obj.metadata.xpath('.//short_description/text()')[0]
        except IndexError:
            sd = 'NoName'
        if iocid in self.iocs:
            msg = 'duplicate IOC UUID [{}] [orig_shortName: {}][new_shortName: {}]'.format(iocid,
                                                                                           self.ioc_name[iocid],
                                                                                           sd)
            log.warning(msg)
        self.iocs[iocid] = ioc_obj
        self.ioc_name[iocid] = sd
        if self.parser_callback:
            self.parser_callback(ioc_obj)
        return True