def _validate_markdown(self, expfile):
        '''ensure that fields are present in markdown file'''

        try:
            import yaml
        except:
            bot.error('Python yaml is required for testing yml/markdown files.')
            sys.exit(1)

        self.metadata = {}
        uid = os.path.basename(expfile).strip('.md')
     
        if os.path.exists(expfile):
            with open(expfile, "r") as stream:
                docs = yaml.load_all(stream)
                for doc in docs:
                    if isinstance(doc,dict):
                        for k,v in doc.items():
                            print('%s: %s' %(k,v))
                            self.metadata[k] = v
            self.metadata['uid'] = uid
       
            fields = ['github', 'preview', 'name', 'layout',
                      'tags', 'uid', 'maintainer']

            # Tests for all fields
            for field in fields:
                if field not in self.metadata:
                    return False
                if self.metadata[field] in ['',None]:
                    return False

            if 'github' not in self.metadata['github']:
                return notvalid('%s: not a valid github repository' % name)
            if not isinstance(self.metadata['tags'],list):
                return notvalid('%s: tags must be a list' % name)
            if not re.search("(\w+://)(.+@)*([\w\d\.]+)(:[\d]+){0,1}/*(.*)", self.metadata['github']):
                return notvalid('%s is not a valid URL.' %(self.metadata['github']))

        return True