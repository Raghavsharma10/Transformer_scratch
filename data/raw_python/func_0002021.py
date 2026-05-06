def convert_to_11(self):
        """
        converts the iocs in self.iocs from openioc 1.0 to openioc 1.1 format.
        the converted iocs are stored in the dictionary self.iocs_11
        """
        if len(self) < 1:
            log.error('No iocs available to modify.')
            return False
        log.info('Converting IOCs from 1.0 to 1.1')
        errors = []
        for iocid in self.iocs:
            ioc_xml = self.iocs[iocid]
            root = ioc_xml.getroot()
            if root.tag != 'ioc':
                log.error('IOC root is not "ioc" [%s].' % str(iocid))
                errors.append(iocid)
                continue
            name_10 = root.findtext('.//short_description')
            keywords_10 = root.findtext('.//keywords')
            description_10 = root.findtext('.//description')
            author_10 = root.findtext('.//authored_by')
            created_date_10 = root.findtext('.//authored_date')
            last_modified_date_10 = root.get('last-modified', None)
            if last_modified_date_10:
                last_modified_date_10 = last_modified_date_10.rstrip('Z')
            created_date_10 = created_date_10.rstrip('Z')
            links_10 = []
            for link in root.xpath('//link'):
                link_rel = link.get('rel', None)
                link_text = link.text
                links_10.append((link_rel, link_text, None))
            # get ioc_logic
            try:
                ioc_logic = root.xpath('.//definition')[0]
            except IndexError:
                log.exception(
                    'Could not find definition nodes for IOC [%s].  Did you attempt to convert OpenIOC 1.1 iocs?' % str(
                        iocid))
                errors.append(iocid)
                continue
            # create 1.1 ioc obj
            ioc_obj = ioc_api.IOC(name=name_10, description=description_10, author=author_10, links=links_10,
                                  keywords=keywords_10, iocid=iocid)
            ioc_obj.set_lastmodified_date(last_modified_date_10)
            ioc_obj.set_created_date(created_date_10)

            comment_dict = {}
            tlo_10 = ioc_logic.getchildren()[0]
            try:
                self.convert_branch(tlo_10, ioc_obj.top_level_indicator, comment_dict)
            except UpgradeError:
                log.exception('Problem converting IOC [{}]'.format(iocid))
                errors.append(iocid)
                continue
            for node_id in comment_dict:
                ioc_obj.add_parameter(node_id, comment_dict[node_id])
            self.iocs_11[iocid] = ioc_obj
        return errors