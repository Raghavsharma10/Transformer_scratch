def convert_to_10(self):
        """
        converts the iocs in self.iocs from openioc 1.1 to openioc 1.0 format.
        the converted iocs are stored in the dictionary self.iocs_10
        :return: A list of iocid values which had errors downgrading.
        """
        if len(self) < 1:
            log.error('no iocs available to modify')
            return False
        log.info('Converting IOCs from 1.1 to 1.0.')
        errors = []
        for iocid in self.iocs:
            pruned = False
            ioc_obj_11 = self.iocs[iocid]
            metadata = ioc_obj_11.metadata
            # record metadata
            name_11 = metadata.findtext('.//short_description')
            keywords_11 = metadata.findtext('.//keywords')
            description_11 = metadata.findtext('.//description')
            author_11 = metadata.findtext('.//authored_by')
            created_date_11 = metadata.findtext('.//authored_date')
            last_modified_date_11 = ioc_obj_11.root.get('last-modified')
            links_11 = []
            for link in metadata.xpath('//link'):
                link_rel = link.get('rel')
                link_text = link.text
                links_11.append((link_rel, None, link_text))
            # get ioc_logic
            try:
                ioc_logic = ioc_obj_11.root.xpath('.//criteria')[0]
            except IndexError:
                log.exception(
                    'Could not find criteria nodes for IOC [{}].  Did you attempt to convert OpenIOC 1.0 iocs?'.format(
                        iocid))
                errors.append(iocid)
                continue
            try:
                tlo_11 = ioc_logic.getchildren()[0]
            except IndexError:
                log.exception(
                    'Could not find children for the top level criteria/children nodes for IOC [{}]'.format(iocid))
                errors.append(iocid)
                continue
            tlo_id = tlo_11.get('id')
            # record comment parameters
            comment_dict = {}
            for param in ioc_obj_11.parameters.xpath('//param[@name="comment"]'):
                param_id = param.get('ref-id')
                param_text = param.findtext('value')
                comment_dict[param_id] = param_text
            # create a 1.1 indicator and populate it with the metadata from the existing 1.1
            # we will then modify this new IOC to conform to 1.1 schema
            ioc_obj_10 = ioc_api.IOC(name=name_11, description=description_11, author=author_11, links=links_11,
                                     keywords=keywords_11, iocid=iocid)
            ioc_obj_10.root.attrib['last-modified'] = last_modified_date_11
            authored_date_node = ioc_obj_10.metadata.find('authored_date')
            authored_date_node.text = created_date_11

            # convert 1.1 ioc object to 1.0
            # change xmlns
            ioc_obj_10.root.attrib['xmlns'] = 'http://schemas.mandiant.com/2010/ioc'
            # remove published data
            del ioc_obj_10.root.attrib['published-date']
            # remove parameters node
            ioc_obj_10.root.remove(ioc_obj_10.parameters)
            # change root tag
            ioc_obj_10.root.tag = 'ioc'
            # metadata underneath the root node
            metadata_node = ioc_obj_10.metadata
            criteria_node = ioc_obj_10.top_level_indicator.getparent()
            metadata_dictionary = {}
            for child in metadata_node:
                metadata_dictionary[child.tag] = child
            for tag in METADATA_REQUIRED_10:
                if tag not in metadata_dictionary:
                    msg = 'IOC {} is missing required metadata: [{}]'.format(iocid, tag)
                    raise DowngradeError(msg)
            for tag in METADATA_ORDER_10:
                if tag in metadata_dictionary:
                    ioc_obj_10.root.append(metadata_dictionary.get(tag))
            ioc_obj_10.root.remove(metadata_node)
            ioc_obj_10.root.remove(criteria_node)
            criteria_node.tag = 'definition'
            ioc_obj_10.root.append(criteria_node)

            ioc_obj_10.top_level_indicator.attrib['id'] = tlo_id
            # identify indicator items with 1.1 specific operators
            # we will skip them when converting IOC from 1.1 to 1.0.
            ids_to_skip = set()
            indicatoritems_to_remove = set()
            for condition_type in self.openioc_11_only_conditions:
                for elem in ioc_logic.xpath('//IndicatorItem[@condition="%s"]' % condition_type):
                    pruned = True
                    indicatoritems_to_remove.add(elem)
            for elem in ioc_logic.xpath('//IndicatorItem[@preserve-case="true"]'):
                pruned = True
                indicatoritems_to_remove.add(elem)
            # walk up from each indicatoritem
            # to build set of ids to skip when downconverting
            for elem in indicatoritems_to_remove:
                nid = None
                current = elem
                while nid != tlo_id:
                    parent = current.getparent()
                    nid = parent.get('id')
                    if nid == tlo_id:
                        current_id = current.get('id')
                        ids_to_skip.add(current_id)
                    else:
                        current = parent
            # walk the 1.1 IOC to convert it into a 1.0 IOC
            # noinspection PyBroadException
            try:
                self.convert_branch(tlo_11, ioc_obj_10.top_level_indicator, ids_to_skip, comment_dict)
            except DowngradeError:
                log.exception('Problem converting IOC [{}]'.format(iocid))
                errors.append(iocid)
                continue
            except Exception:
                log.exception('Unknown error occured while converting [{}]'.format(iocid))
                errors.append(iocid)
                continue
            # bucket pruned iocs / null iocs
            if not ioc_obj_10.top_level_indicator.getchildren():
                self.null_pruned_iocs.add(iocid)
            elif pruned is True:
                self.pruned_11_iocs.add(iocid)
            # Check the original to see if there was a comment prior to the root node, and if so, copy it's content
            comment_node = ioc_obj_11.root.getprevious()
            while comment_node is not None:
                log.debug('found a comment node')
                c = et.Comment(comment_node.text)
                ioc_obj_10.root.addprevious(c)
                comment_node = comment_node.getprevious()
            # Record the IOC
            # ioc_10 = et.ElementTree(root_10)
            self.iocs_10[iocid] = ioc_obj_10
        return errors