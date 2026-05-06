def acquire_metadata(self):
        """
        Handles the acquisition of metadata for both collection mode and single
        mode, uses the metadata methods belonging to the article's publisher
        attribute.
        """
        #For space economy
        publisher = self.article.publisher

        if self.collection:  # collection mode metadata gathering
            pass
        else:  # single mode metadata gathering
            self.pub_id = publisher.package_identifier()
            self.title = publisher.package_title()
            for date in publisher.package_date():
                self.dates.add(date)

        #Common metadata gathering
        for lang in publisher.package_language():
            self.languages.add(lang)  # languages
        for contributor in publisher.package_contributors():  # contributors
            self.contributors.add(contributor)
        self.publishers.add(publisher.package_publisher())  # publisher names
        desc = publisher.package_description()
        if desc is not None:
            self.descriptions.add(desc)
        for subj in publisher.package_subject():
            self.subjects.add(subj)  # subjects
        #Rights
        art_rights = publisher.package_rights()
        self.rights.add(art_rights)
        if art_rights not in self.rights_associations:
            self.rights_associations[art_rights] = [self.article.doi]
        else:
            self.rights_associations[art_rights].append(self.article.doi)