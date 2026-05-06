def appendText(self, text):
        '''
            appendText - append some inner text
        '''
        # self.text is just raw string of the text
        self.text += text
        self.isSelfClosing = False # inner text means it can't self close anymo
        # self.blocks is either text or tags, in order of appearance
        self.blocks.append(text)