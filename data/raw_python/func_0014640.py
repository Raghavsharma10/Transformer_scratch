def find(self, **kwargs):
        '''
            find - Perform a search of elements using attributes as keys and potential values as values
            
               (i.e.  parser.find(name='blah', tagname='span')  will return all elements in this document
                 with the name "blah" of the tag type "span" )

            Arguments are key = value, or key can equal a tuple/list of values to match ANY of those values.

            Append a key with __contains to test if some strs (or several possible strs) are within an element
            Append a key with __icontains to perform the same __contains op, but ignoring case

            Special keys:

               tagname    - The tag name of the element
               text       - The text within an element

            NOTE: Empty string means both "not set" and "no value" in this implementation.

            NOTE: If you installed the QueryableList module (i.e. ran setup.py without --no-deps) it is
              better to use the "filter"/"filterAnd" or "filterOr" methods, which are also available
              on all tags and tag collections (tag collections also have filterAllAnd and filterAllOr)


            @return TagCollection<AdvancedTag> - A list of tags that matched the filter criteria
        '''

        if not kwargs:
            return TagCollection()


        # Because of how closures work in python, need a function to generate these lambdas
        #  because the closure basically references "current key in iteration" and not
        #  "actual instance" of variable. Seems to me to be a bug... but whatever
        def _makeTagnameLambda(tagName):
            return lambda em : em.tagName == tagName

        def _makeAttributeLambda(_key, _value):
            return lambda em : em.getAttribute(_key, '') == _value

        def _makeTagnameInLambda(tagNames):
            return lambda em : em.tagName in tagNames

        def _makeAttributeInLambda(_key, _values):
            return lambda em : em.getAttribute(_key, '') in _values

        def _makeTextLambda(_value):
            return lambda em : em.text == _value

        def _makeTextInLambda(_values):
            return lambda em : em.text in _values

        def _makeAttributeContainsLambda(_key, _value, icontains=False):
            if icontains is False:
                return lambda em : _value in em.getAttribute(_key, '')
            else:
                _value = _value.lower()
                return lambda em : _value in em.getAttribute(_key, '').lower()

        def _makeTextContainsLambda(_value, icontains=False):
            if icontains is False:
                return lambda em : _value in em.text
            else:
                _value = _value.lower()
                return lambda em : _value in em.text.lower()

        def _makeAttributeContainsInLambda(_key, _values, icontains=False):
            if icontains:
                _values = tuple([x.lower() for x in _values])

            def _testFunc(em):
                attrValue = em.getAttribute(_key, '')
                if icontains:
                    attrValue = attrValue.lower()

                for value in _values:
                    if value in attrValue:
                        return True

                return False

            return _testFunc

        def _makeTextContainsInLambda(_values, icontains=False):
            if icontains:
                _values = tuple([x.lower() for x in _values])

            def _testFunc(em):
                text = em.text
                if icontains:
                    text = text.lower()

                for value in _values:
                    if value in text:
                        return True

                return False

            return _testFunc

        # This will hold all the functions we will chain for matching
        matchFunctions = []

        # Iterate over all the filter portions, and build a filter.
        for key, value in kwargs.items():
            key = key.lower()

            endsIContains = key.endswith('__icontains')
            endsContains = key.endswith('__contains')

            isValueList = isinstance(value, (list, tuple))

            thisFunc = None

            if endsIContains or endsContains:
                key = re.sub('__[i]{0,1}contains$', '', key)
                if key == 'tagname':
                    raise ValueError('tagname is not supported for contains')

                if isValueList:
                    if key == 'text':
                        thisFunc = _makeTextContainsInLambda(value, icontains=endsIContains)
                    else:
                        thisFunc = _makeAttributeContainsLambda(key, value, icontains=endsIContains)
                else:
                    if key == 'text':
                        thisFunc = _makeTextContainsLambda(value, icontains=endsIContains)
                    else:
                        thisFunc = _makeAttributeContainsLambda(key, value, icontains=endsIContains)

            else:
                # Not contains, straight up

                if isValueList:
                    if key == 'tagname':
                        thisFunc = _makeTagnameInLambda(value)
                    elif key == 'text':
                        thisFunc = _makeTextInLambda(value)
                    else:
                        thisFunc = _makeAttributeInLambda(key, value)
                else:
                    if key == 'tagname':
                        thisFunc = _makeTagnameLambda(value)
                    elif key == 'text':
                        thisFunc = _makeTextLambda(value)
                    else:
                        thisFunc = _makeAttributeLambda(key, value)


            matchFunctions.append( thisFunc )

        # The actual matching function - This will run through the assembled
        #  #matchFunctions list, testing the element against each match
        #  and returning all elements in a TagCollection that match this list.
        def doMatchFunc(em):
            for matchFunction in matchFunctions:
                if matchFunction(em) is False:
                    return False

            return True

        return self.getElementsCustomFilter(doMatchFunc)