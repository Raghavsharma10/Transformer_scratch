def get_edxml_with_aws_urls(self):
        """stub"""
        edxml = self.get_edxml()
        soup = BeautifulSoup(edxml, 'xml')

        attrs = {
            'draggable': 'icon',
            'drag_and_drop_input': 'img',
            'files': 'included_files',
            'img': 'src'
        }
        # replace all file listings with an appropriate path...
        if len(self.my_osid_object.object_map['fileIds']) > 0:
            file_map = self.my_osid_object.get_files()
            for file_label, url in file_map.items():
                local_regex = re.compile(file_label + r'\.')
                for key, attr in attrs.items():
                    search = {attr: local_regex}
                    tags = soup.find_all(**search)
                    for item in tags:
                        item[attr] = url
        return soup.find('problem').prettify()