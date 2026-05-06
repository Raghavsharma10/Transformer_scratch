def clear_text(self, *args, **selectors):
        """
        Clear text of the UI object  with *selectors*
        """
        while True:
            target = self.device(**selectors)
            text = target.info['text']
            target.clear_text()
            remain_text = target.info['text']
            if text == ''  or remain_text == text:
                break