def move_back_boxed_texts(self):
        """
        The only intended use for this function is to patch a problem seen in
        at least one PLoS article (journal.pgen.0020002). This will move any
        <boxed-text> elements over to the receiving element, which is probably
        the main body.
        """
        body = self.main.getroot().find('body')
        back = self.article.root.find('back')
        if back is None:
            return
        boxed_texts = back.xpath('.//boxed-text')
        for boxed_text in boxed_texts:
            body.append(deepcopy(boxed_text))