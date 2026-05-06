def get_description(self):
        """
        Tries to get WF description from 'collabration' or 'process' or 'pariticipant'

        Returns str: WF description

        """
        paths = ['bpmn:collaboration/bpmn:participant/bpmn:documentation',
                 'bpmn:collaboration/bpmn:documentation',
                 'bpmn:process/bpmn:documentation']
        for path in paths:
            elm = self.root.find(path, NS)
            if elm is not None and elm.text:
                return elm.text