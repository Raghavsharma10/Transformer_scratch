def get_name(self):
        """
        Tries to get WF name from 'process' or 'collobration' or 'pariticipant'

        Returns:
            str. WF name.
        """
        ns = {'ns': '{%s}' % BPMN_MODEL_NS}
        for path in ('.//{ns}process',
                     './/{ns}collaboration',
                     './/{ns}collaboration/{ns}participant/'):
            tag = self.doc_xpath(path.format(**ns))
            if tag:
                name = tag[0].get('name')
                if name:
                    return name
        return self.get_id()