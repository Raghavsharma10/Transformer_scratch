def getStartTag(self, *args, **kwargs):
        '''
            getStartTag - Override the end-spacing rules

              @see AdvancedTag.getStartTag
        '''

        ret = AdvancedTag.getStartTag(self, *args, **kwargs)

        if ret.endswith(' >'):
            ret = ret[:-2] + '>'
        elif object.__getattribute__(self, 'slimSelfClosing') and ret.endswith(' />'):
            ret = ret[:-3] + '/>'

        return ret