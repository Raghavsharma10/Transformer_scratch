def add_python_plugin(self, name, python_file, value_type="gauge", unit=None):
        """
        Add a custom python plugin to the collectd instance of a monitored plugin

        :param python_file: Path of the python file to be added as custom python plugin
        :param name: Name of the plugin
        :param value_type: Optional. Can be either "gauge" or "derive"
        :param unit: Optional. If given the new plugin will be measured according to this unit
        """

        if not os.path.isfile(python_file):
            raise Exception(python_file, "is not a file or could not be found in tho given path")

        with open(python_file) as f:
            script = f.read()

        payload = {
            'plugin_type': 'python',
            'name': name,
            'unit': unit,
            'value_type': value_type,
            'read_function': script,
            'host': self.info['public_ips'][0]
        }

        data = json.dumps(payload)

        #PLugin id must be in lowercase
        plugin_id = name.lower()

        #PLugin id must contain only alphanumeric chars
        pattern = re.compile('\W')
        plugin_id = re.sub(pattern, "_", plugin_id)

        #Plugin id should not have double underscores
        while "__" in plugin_id:
            pattern = "\r?__"
            plugin_id = re.sub(pattern, "_", plugin_id)

        #Plugin id should not have underscore as first or last char
        if plugin_id[-1] == "_":
            plugin_id = plugin_id[:-2]

        if plugin_id[0] == "_":
            plugin_id = plugin_id[1:]

        req = self.request(self.mist_client.uri+"/clouds/"+self.cloud.id+"/machines/"+self.id+"/plugins/"+plugin_id,
                           data=data)
        req.post()