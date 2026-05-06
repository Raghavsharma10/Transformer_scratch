def _get_lane_properties(self, node):
        """
        Parses the given XML node

        Args:
            node (xml): XML node.

        .. code-block:: xml

             <bpmn2:lane id="Lane_8" name="Lane 8">
                <bpmn2:extensionElements>
                    <camunda:properties>
                        <camunda:property value="foo,bar" name="perms"/>
                    </camunda:properties>
                </bpmn2:extensionElements>
            </bpmn2:lane>

        Returns:
            {'perms': 'foo,bar'}
        """
        lane_name = self.get_lane(node.get('id'))
        lane_data = {'name': lane_name}
        for a in self.xpath(".//bpmn:lane[@name='%s']/*/*/" % lane_name):
            lane_data[a.attrib['name']] = a.attrib['value'].strip()
        return lane_data