def scroll_to_vertically(self, obj, *args,**selectors):
        """
        Scroll(vertically) on the object: obj to specific UI object which has *selectors* attributes appears.

        Return true if the UI object, else return false.

        Example:

        | ${list}        | Get Object           | className=android.widget.ListView |              | # Get the list object     |
        | ${is_web_view} | Scroll To Vertically | ${list}                           | text=WebView | # Scroll to text:WebView. |
        """
        return obj.scroll.vert.to(**selectors)