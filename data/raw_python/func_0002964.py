def make_back_matter(self):
        """
        The <back> element may have 0 or 1 <label> elements and 0 or 1 <title>
        elements. Then it may have any combination of the following: <ack>,
        <app-group>, <bio>, <fn-group>, <glossary>, <ref-list>, <notes>, and
        <sec>. <sec> is employed here as a catch-all for material that does not
        fall under the other categories.

        The Back should generally be thought of as a non-linear element, though
        some of its content will be parsed to the linear flow of the document.
        This can be thought of as critically important meta-information that
        should accompany the main text (e.g. Acknowledgments and Contributions)

        Because the content of <back> contains a set of tags that intersects
        with that of the Body, this method should always be called before the
        general post-processing steps; keep in mind that this is also the
        opportunity to permit special handling of content in the Back
        """

        #Back is technically metadata content that needs to be interpreted to
        #presentable content
        body = self.main.getroot().find('body')
        if self.article.root.find('back') is None:
            return
        #The following things are ordered in such a way to adhere to what
        #appears to be a consistent presentation order for PLoS
        #Acknowledgments
        back_ack = self.make_back_acknowledgments()
        if back_ack is not None:
            body.append(back_ack)
        #Author Contributions
        self.make_back_author_contributions(body)
        #Glossaries
        self.make_back_glossary(body)
        #Notes
        self.make_back_notes(body)