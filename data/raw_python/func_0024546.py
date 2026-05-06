def get_name(self):
        """
        Tries to get WF name from 'process' or 'collobration' or 'pariticipant'

        Returns:
            str. WF name.
        """
        paths = ['bpmn:process',
                 'bpmn:collaboration/bpmn:participant/',
                 'bpmn:collaboration',
                 ]
        for path in paths:
            tag = self.root.find(path, NS)
            if tag is not None and len(tag):
                name = tag.get('name')
                if name:
                    return name