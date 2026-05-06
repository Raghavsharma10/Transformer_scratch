def trill_links(self):
        """dict: trill link details
        """
        xmlns = 'urn:brocade.com:mgmt:brocade-fabric-service'
        get_links_info = ET.Element('show-linkinfo', xmlns=xmlns)
        results = self._callback(get_links_info, handler='get')
        result = []
        for item in results.findall('{%s}show-link-info' % xmlns):
            src_rbridge_id = item.find('{%s}linkinfo-rbridgeid' % xmlns).text
            src_switch_wwn = item.find('{%s}linkinfo-wwn' % xmlns).text
            for link in item.findall('{%s}linkinfo-isl' % xmlns):
                dest_rbridge_id = link.find(
                    '{%s}linkinfo-isllink-destdomain' % xmlns).text
                src_interface = link.find(
                    '{%s}linkinfo-isllink-srcport-interface' % xmlns).text
                dest_interface = link.find(
                    '{%s}linkinfo-isllink-destport-interface' % xmlns).text
                link_cost = link.find('{%s}linkinfo-isl-linkcost' % xmlns).text
                link_cost_count = link.find(
                    '{%s}linkinfo-isllink-costcount' % xmlns).text

                item_results = {'source-rbridgeid': src_rbridge_id,
                                'source-switch-wwn': src_switch_wwn,
                                'dest-rbridgeid': dest_rbridge_id,
                                'source-interface': src_interface,
                                'dest-interface': dest_interface,
                                'link-cost': link_cost,
                                'link-costcount': link_cost_count}

                result.append(item_results)

        return result