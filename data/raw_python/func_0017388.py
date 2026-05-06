def get(self):
        """Return form result"""
        # It is import to avoid accessing Qt C++ object as it has probably
        # already been destroyed, due to the Qt.WA_DeleteOnClose attribute
        if self.outfile:
            if self.result in ['list', 'dict', 'OrderedDict']:
                fd = open(self.outfile + '.py', 'w')
                fd.write(str(self.data))
            elif self.result == 'JSON':
                fd = open(self.outfile + '.json', 'w')
                data = json.loads(self.data, object_pairs_hook=OrderedDict)
                json.dump(data, fd)
            elif self.result == 'XML':
                fd = open(self.outfile + '.xml', 'w')
                root = ET.fromstring(self.data)
                tree = ET.ElementTree(root)
                tree.write(fd, encoding='UTF-8')
            fd.close()
        else:
            return self.data