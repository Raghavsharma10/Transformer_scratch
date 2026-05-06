def feed(self, contents):
        '''
            feed - Feed contents. Use  parseStr or parseFile instead.

            @param contents - Contents
        '''
        contents = stripIEConditionals(contents)
        try:
            HTMLParser.feed(self, contents)
        except MultipleRootNodeException:
            self.reset()
            HTMLParser.feed(self, "%s%s" %(addStartTag(contents, INVISIBLE_ROOT_TAG_START), INVISIBLE_ROOT_TAG_END))