def dump(self, f, indent=''):
        """Dump this section and its children to a file-like object"""
        print(("%s&%s %s" % (indent, self.__name, self.section_parameters)).rstrip(), file=f)
        self.dump_children(f, indent)
        print("%s&END %s" % (indent, self.__name), file=f)