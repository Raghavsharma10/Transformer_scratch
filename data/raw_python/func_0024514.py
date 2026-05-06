def _get_description(self):
        """
        Tries to get WF description from 'collabration' or 'process' or 'pariticipant'
        Returns:

        """
        ns = {'ns': '{%s}' % BPMN_MODEL_NS}
        desc = (
            self.doc_xpath('.//{ns}collaboration/{ns}documentation'.format(**ns)) or
            self.doc_xpath('.//{ns}process/{ns}documentation'.format(**ns)) or
            self.doc_xpath('.//{ns}collaboration/{ns}participant/{ns}documentation'.format(**ns))
        )
        if desc:
            return desc[0].findtext('.')