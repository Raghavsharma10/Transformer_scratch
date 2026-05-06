def _parse_clublog_xml(self, cty_xml_filename):
        """
        parse the content of a clublog XML file and return the
        parsed values in dictionaries

        """

        entities = {}
        call_exceptions = {}
        prefixes = {}
        invalid_operations = {}
        zone_exceptions = {}

        call_exceptions_index = {}
        prefixes_index = {}
        invalid_operations_index = {}
        zone_exceptions_index = {}

        cty_tree = ET.parse(cty_xml_filename)
        root = cty_tree.getroot()

        #retrieve ADIF Country Entities
        cty_entities = cty_tree.find("entities")
        self._logger.debug("total entities: " + str(len(cty_entities)))
        if len(cty_entities) > 1:
            for cty_entity in cty_entities:
                try:
                    entity = {}
                    for item in cty_entity:
                        if item.tag == "name":
                            entity[const.COUNTRY] = unicode(item.text)
                            self._logger.debug(unicode(item.text))
                        elif item.tag == "prefix":
                            entity[const.PREFIX] = unicode(item.text)
                        elif item.tag == "deleted":
                            if item.text == "TRUE":
                                entity[const.DELETED] = True
                            else:
                                entity[const.DELETED] = False
                        elif item.tag == "cqz":
                            entity[const.CQZ] = int(item.text)
                        elif item.tag == "cont":
                            entity[const.CONTINENT] = unicode(item.text)
                        elif item.tag == "long":
                            entity[const.LONGITUDE] = float(item.text)
                        elif item.tag == "lat":
                            entity[const.LATITUDE] = float(item.text)
                        elif item.tag == "start":
                            dt = datetime.strptime(item.text[:19], '%Y-%m-%dT%H:%M:%S')
                            entity[const.START] = dt.replace(tzinfo=UTC)
                        elif item.tag == "end":
                            dt = datetime.strptime(item.text[:19], '%Y-%m-%dT%H:%M:%S')
                            entity[const.END] = dt.replace(tzinfo=UTC)
                        elif item.tag == "whitelist":
                            if item.text == "TRUE":
                                entity[const.WHITELIST] = True
                            else:
                                entity[const.WHITELIST] = False
                        elif item.tag == "whitelist_start":
                            dt = datetime.strptime(item.text[:19], '%Y-%m-%dT%H:%M:%S')
                            entity[const.WHITELIST_START] = dt.replace(tzinfo=UTC)
                        elif item.tag == "whitelist_end":
                            dt = datetime.strptime(item.text[:19], '%Y-%m-%dT%H:%M:%S')
                            entity[const.WHITELIST_END] = dt.replace(tzinfo=UTC)
                except AttributeError:
                    self._logger.error("Error while processing: ")
                entities[int(cty_entity[0].text)] = entity
            self._logger.debug(str(len(entities))+" Entities added")
        else:
            raise Exception("No Country Entities detected in XML File")


        cty_exceptions = cty_tree.find("exceptions")
        if len(cty_exceptions) > 1:
            for cty_exception in cty_exceptions:
                call_exception = {}
                for item in cty_exception:
                    if item.tag == "call":
                        call = str(item.text)
                        if call in call_exceptions_index.keys():
                            call_exceptions_index[call].append(int(cty_exception.attrib["record"]))
                        else:
                            call_exceptions_index[call] = [int(cty_exception.attrib["record"])]
                    elif item.tag == "entity":
                        call_exception[const.COUNTRY] = unicode(item.text)
                    elif item.tag == "adif":
                        call_exception[const.ADIF] = int(item.text)
                    elif item.tag == "cqz":
                        call_exception[const.CQZ] = int(item.text)
                    elif item.tag == "cont":
                        call_exception[const.CONTINENT] = unicode(item.text)
                    elif item.tag == "long":
                        call_exception[const.LONGITUDE] = float(item.text)
                    elif item.tag == "lat":
                        call_exception[const.LATITUDE] = float(item.text)
                    elif item.tag == "start":
                        dt = datetime.strptime(item.text[:19], '%Y-%m-%dT%H:%M:%S')
                        call_exception[const.START] = dt.replace(tzinfo=UTC)
                    elif item.tag == "end":
                        dt = datetime.strptime(item.text[:19], '%Y-%m-%dT%H:%M:%S')
                        call_exception[const.END] = dt.replace(tzinfo=UTC)
                    call_exceptions[int(cty_exception.attrib["record"])] = call_exception

            self._logger.debug(str(len(call_exceptions))+" Exceptions added")
            self._logger.debug(str(len(call_exceptions_index))+" unique Calls in Index ")

        else:
            raise Exception("No Exceptions detected in XML File")


        cty_prefixes = cty_tree.find("prefixes")
        if len(cty_prefixes) > 1:
            for cty_prefix in cty_prefixes:
                prefix = {}
                for item in cty_prefix:
                    pref = None
                    if item.tag == "call":

                        #create index for this prefix
                        call = str(item.text)
                        if call in prefixes_index.keys():
                            prefixes_index[call].append(int(cty_prefix.attrib["record"]))
                        else:
                            prefixes_index[call] = [int(cty_prefix.attrib["record"])]
                    if item.tag == "entity":
                        prefix[const.COUNTRY] = unicode(item.text)
                    elif item.tag == "adif":
                        prefix[const.ADIF] = int(item.text)
                    elif item.tag == "cqz":
                        prefix[const.CQZ] = int(item.text)
                    elif item.tag == "cont":
                        prefix[const.CONTINENT] = unicode(item.text)
                    elif item.tag == "long":
                        prefix[const.LONGITUDE] = float(item.text)
                    elif item.tag == "lat":
                        prefix[const.LATITUDE] = float(item.text)
                    elif item.tag == "start":
                        dt = datetime.strptime(item.text[:19], '%Y-%m-%dT%H:%M:%S')
                        prefix[const.START] = dt.replace(tzinfo=UTC)
                    elif item.tag == "end":
                        dt = datetime.strptime(item.text[:19], '%Y-%m-%dT%H:%M:%S')
                        prefix[const.END] = dt.replace(tzinfo=UTC)
                    prefixes[int(cty_prefix.attrib["record"])] = prefix

            self._logger.debug(str(len(prefixes))+" Prefixes added")
            self._logger.debug(str(len(prefixes_index))+" unique Prefixes in Index")
        else:
            raise Exception("No Prefixes detected in XML File")

        cty_inv_operations = cty_tree.find("invalid_operations")
        if len(cty_inv_operations) > 1:
            for cty_inv_operation in cty_inv_operations:
                invalid_operation = {}
                for item in cty_inv_operation:
                    call = None
                    if item.tag == "call":
                        call = str(item.text)
                        if call in invalid_operations_index.keys():
                            invalid_operations_index[call].append(int(cty_inv_operation.attrib["record"]))
                        else:
                            invalid_operations_index[call] = [int(cty_inv_operation.attrib["record"])]

                    elif item.tag == "start":
                        dt = datetime.strptime(item.text[:19], '%Y-%m-%dT%H:%M:%S')
                        invalid_operation[const.START] = dt.replace(tzinfo=UTC)
                    elif item.tag == "end":
                        dt = datetime.strptime(item.text[:19], '%Y-%m-%dT%H:%M:%S')
                        invalid_operation[const.END] = dt.replace(tzinfo=UTC)
                    invalid_operations[int(cty_inv_operation.attrib["record"])] = invalid_operation

            self._logger.debug(str(len(invalid_operations))+" Invalid Operations added")
            self._logger.debug(str(len(invalid_operations_index))+" unique Calls in Index")
        else:
            raise Exception("No records for invalid operations detected in XML File")


        cty_zone_exceptions = cty_tree.find("zone_exceptions")
        if len(cty_zone_exceptions) > 1:
            for cty_zone_exception in cty_zone_exceptions:
                zoneException = {}
                for item in cty_zone_exception:
                    call = None
                    if item.tag == "call":
                        call = str(item.text)
                        if call in zone_exceptions_index.keys():
                            zone_exceptions_index[call].append(int(cty_zone_exception.attrib["record"]))
                        else:
                            zone_exceptions_index[call] = [int(cty_zone_exception.attrib["record"])]

                    elif item.tag == "zone":
                        zoneException[const.CQZ] = int(item.text)
                    elif item.tag == "start":
                        dt = datetime.strptime(item.text[:19], '%Y-%m-%dT%H:%M:%S')
                        zoneException[const.START] = dt.replace(tzinfo=UTC)
                    elif item.tag == "end":
                        dt = datetime.strptime(item.text[:19], '%Y-%m-%dT%H:%M:%S')
                        zoneException[const.END] = dt.replace(tzinfo=UTC)
                    zone_exceptions[int(cty_zone_exception.attrib["record"])] = zoneException

            self._logger.debug(str(len(zone_exceptions))+" Zone Exceptions added")
            self._logger.debug(str(len(zone_exceptions_index))+" unique Calls in Index")
        else:
            raise Exception("No records for zone exceptions detected in XML File")

        result = {
            "entities" : entities,
            "call_exceptions" : call_exceptions,
            "prefixes" : prefixes,
            "invalid_operations" : invalid_operations,
            "zone_exceptions" : zone_exceptions,
            "prefixes_index" : prefixes_index,
            "call_exceptions_index" : call_exceptions_index,
            "invalid_operations_index" : invalid_operations_index,
            "zone_exceptions_index" : zone_exceptions_index,
        }
        return result