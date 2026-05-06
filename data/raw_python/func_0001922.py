def get_embedded_yara(self, iocid):
        """
        Extract YARA signatures embedded in Yara/Yara indicatorItem nodes.
        This is done regardless of logic structure in the OpenIOC.
        """
        ioc_obj = self.iocs[iocid]
        ids_to_process = set([])
        signatures = ''
        for elem in ioc_obj.top_level_indicator.xpath('.//IndicatorItem[Context/@search = "Yara/Yara"]'):
            signature = elem.findtext('Content')
            signatures = signatures + '\n' + signature
        if signatures:
            signatures += '\n'
        return signatures