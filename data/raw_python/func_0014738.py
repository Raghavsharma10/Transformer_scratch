def handle_data(self, data):
        '''
            handle_data - Internal for parsing
        '''
        if data:
            inTag = self._inTag
            if len(inTag) > 0:
                if inTag[-1].tagName not in PRESERVE_CONTENTS_TAGS:
                    data = data.replace('\t', ' ').strip('\r\n')
                    if data.startswith(' '):
                        data = ' ' + data.lstrip()
                    if data.endswith(' '):
                        data = data.rstrip() + ' '
                inTag[-1].appendText(data)
            elif data.strip():
                # Must be text prior to or after root node
                raise MultipleRootNodeException()