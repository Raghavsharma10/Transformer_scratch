def get_yara_condition_string(self, indicator_node, parameters_node, ids_to_process, condition_string='',
                                  joining_value='or'):
        """
        get_yara_condition_string

        input
            indicator_node: this is the node we walk down
            parameters_node: this contains all the parameters in the ioc, so we
                can look up parameters nodes as we walk them.
            ids_to_process: set of ids to upgrade
            condition_string: This represnts the yara condition string.  This
                string grows as we walk nodes.
        return
            returns True upon completion
            may raise ValueError
        """

        indicator_node_id = str(indicator_node.get('id'))
        if indicator_node_id not in ids_to_process:
            msg = 'Entered into get_yara_condition_string with a invalid node to walk [[}]'.format(indicator_node_id)
            raise YaraConversionError(msg)
        expected_tag = 'Indicator'
        if indicator_node.tag != expected_tag:
            raise YaraConversionError('indicator_node expected tag is [%s]' % expected_tag)
        is_set = None
        # print 'indicator node id [%s]' % str(indicator_node_id)
        for param in parameters_node.xpath('.//param[@ref-id="{}"]'.format(indicator_node_id)):
            if param.attrib['name'] == 'yara/set':
                is_set = True
                set_count = param.findtext('value', None)
                try:
                    temp = int(set_count)
                    if temp < 1:
                        raise YaraConversionError('yara/set parameter value was less than 1')
                    if temp > len(indicator_node.getchildren()):
                        msg = 'yara/set value is greater than the number of children' \
                              ' of Indicator node [%s]' % str(indicator_node_id)
                        raise YaraConversionError(msg)
                except ValueError:
                    raise YaraConversionError('yara/set parameter was not a integer')
                set_dict = {'set_count': set_count, 'set_ids': []}

        for node in indicator_node.getchildren():
            node_id = node.get('id')
            # XXX strip out '-' characters from the ids.  If a guid is used as
            # the id, this will cause processing errors
            safe_node_id = node_id.replace('-', '')
            if node_id not in ids_to_process:
                continue
            if node.tag == 'IndicatorItem':
                # print 'handling indicatoritem: [%s]' % node_id
                if is_set:
                    set_dict['set_ids'].append('$' + safe_node_id)
                else:
                    # Default mapping
                    mapping = {'prefix': '$', 'identifier': safe_node_id, 'condition': '', 'postfix': ''}
                    # XXX: Alot of this could raise ValueError
                    use_condition_template = False
                    negation = node.get('negate')
                    condition = node.get('condition')
                    search = node.xpath('Context/@search')[0]
                    content = node.findtext('Content')

                    yara_condition = self.condition_to_yara_map[condition]
                    if not yara_condition:
                        msg = 'Invalid IndicatorItem condition encountered [%s][%s]' % (str(node_id), str(condition))
                        raise YaraConversionError(msg)
                    if negation.lower() == 'true':
                        negation = True
                    else:
                        negation = False
                    # parameters cannot modifier the condition of FileSize or Rule
                    if search == 'Yara/FileSize':
                        mapping['prefix'] = ''
                        mapping['identifier'] = 'filesize'
                        mapping['postfix'] = ' ' + content
                        mapping['condition'] = yara_condition
                        use_condition_template = True
                    elif search == 'Yara/RuleName':
                        if content not in self.ioc_names_set:
                            if mangle_name(content) in self.ioc_names_mangled_set:
                                msg = 'Yara/RuleName is present as a mangled name.[{}][{}]'.format(mangle_name(content),
                                                                                                   node_id)
                                log.warning(msg)
                                content = mangle_name(content)
                            else:
                                log.warning('Yara/RuleName points to a name [{}] that is not in the set of IOCs being'
                                            ' processed [{}]'.format(content, node_id))
                        if mangle_name(content) != content:
                            msg = 'Yara/RuleName contains characters which would cause libyara errors' \
                                  ' [{}]'.format(node_id)
                            raise YaraConversionError(msg)
                        mapping['prefix'] = ''
                        mapping['identifier'] = content
                    # handle parameters
                    else:
                        xp = './/param[@ref-id="{}" and (@name="yara/count" or @name="yara/offset/at" or' \
                             ' @name="yara/offset/in")]'.format(node_id)
                        params = parameters_node.xpath(xp)
                        if len(params) > 1:
                            msg = 'More than one condition parameters assigned to IndicatorItem [{}]'.format(node_id)
                            raise YaraConversionError(msg)
                        for param in params:
                            param_name = param.get('name', None)
                            if param_name == 'yara/count':
                                log.debug('Found [%s] attached to [%s]' % (param.attrib['name'], node_id))
                                mapping['prefix'] = '#'
                                mapping['postfix'] = ' ' + param.findtext('value')
                                mapping['condition'] = yara_condition
                                use_condition_template = True
                                break
                            elif param_name == 'yara/offset/at':
                                log.debug('Found [%s] attached to [%s]' % (param.attrib['name'], node_id))
                                mapping['condition'] = 'at'
                                mapping['postfix'] = ' ' + param.findtext('value')
                                use_condition_template = True
                                break
                            elif param_name == 'yara/offset/in':
                                log.debug('Found [%s] attached to [%s]' % (param.attrib['name'], node_id))
                                mapping['condition'] = 'in'
                                mapping['postfix'] = ' ' + param.findtext('value')
                                use_condition_template = True
                                break

                    if use_condition_template:
                        temp_string = self.yara_II_condition_template % mapping
                    else:
                        temp_string = self.yara_II_template % mapping

                    if condition_string == '':
                        condition_string = temp_string
                    else:
                        condition_string = ' '.join([condition_string, joining_value, temp_string])
                        # print condition_string

            elif node.tag == 'Indicator':
                if is_set:
                    raise YaraConversionError('Cannot have Indicator nodes underneath a Indicator node with yara/set')
                operator = node.get('operator').lower()
                if operator not in ['or', 'and']:
                    raise YaraConversionError('Indicator@operator is not and/or. [%s] has [%s]' % (id, operator))
                # handle parameters
                # XXX Temp POC
                recursed_condition = self.get_yara_condition_string(node, parameters_node, ids_to_process, '', operator)
                xp = './/param[@ref-id="{}" and @name="yara/set"]'.format(node_id)
                if (not parameters_node.xpath(xp)) and has_siblings(node):
                    recursed_condition = '(%s)' % recursed_condition

                if condition_string == '':
                    condition_string = recursed_condition
                else:
                    condition_string = ' '.join([condition_string, joining_value, recursed_condition])
                    # print 'recursed got: [%s]' % condition_string
            else:
                # should never get here
                raise YaraConversionError('node.tag is not a Indicator/IndicatorItem [%s]' % str(id))

        if is_set:
            log.debug('Building set expression for [%s]' % indicator_node_id)
            if len(set_dict['set_ids']) == 0:
                raise YaraConversionError('yara/set processing did not yield any set ids')
            elif len(set_dict['set_ids']) == 1:
                log.warning('yara/set with 1 id found for node [%s]' % node_id)
                set_ids = ''.join(set_dict['set_ids'])
            else:
                set_ids = ','.join(set_dict['set_ids'])
            set_dict['set_ids'] = set_ids
            temp_set_string = self.yara_set_string_template % set_dict
            # print temp_set_string
            if condition_string == '':
                condition_string = temp_set_string
            else:
                condition_string = ' '.join(
                    [condition_string, indicator_node.getparent().get('operator').lower(), temp_set_string])

        return condition_string