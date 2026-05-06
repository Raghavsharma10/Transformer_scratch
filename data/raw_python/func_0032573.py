def child_static(self, context):
        """
        Serve a container page for static content for Mantissa and other
        offerings.
        """
        offeringTech = IOfferingTechnician(self.siteStore)
        installedOfferings = offeringTech.getInstalledOfferings()
        offeringsWithContent = dict([
                (offering.name, offering.staticContentPath)
                for offering
                in installedOfferings.itervalues()
                if offering.staticContentPath])

        # If you wanted to do CSS rewriting for all CSS files served beneath
        # /static/, you could do it by passing a processor for ".css" here.
        # eg:
        #
        # website = IResource(self.store)
        # factory = StylesheetFactory(
        #     offeringsWithContent.keys(), website.rootURL)
        # StaticContent(offeringsWithContent, {
        #               ".css": factory.makeStylesheetResource})
        return StaticContent(offeringsWithContent, {})