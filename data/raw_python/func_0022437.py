def get_info_of_object(self, obj, selector=None):
        """
        return info dictionary of the *obj*
        The info example:
        {
         u'contentDescription': u'',
         u'checked': False,
         u'scrollable': True,
         u'text': u'',
         u'packageName': u'com.android.launcher',
         u'selected': False,
         u'enabled': True,
         u'bounds': 
                   {
                    u'top': 231,
                    u'left': 0,
                    u'right': 1080,
                    u'bottom': 1776
                   },
         u'className': u'android.view.View',
         u'focusable': False,
         u'focused': False,
         u'clickable': False,
         u'checkable': False,
         u'chileCount': 1,
         u'longClickable': False,
         u'visibleBounds':
                          {
                           u'top': 231,
                           u'left': 0,
                           u'right': 1080,
                           u'bottom': 1776
                          }
        }
        """
        if selector:
            return obj.info.get(selector)
        else:
            return obj.info