def extractValue(self, model, item):
        """
        Get the path referenced by this column's attribute.

        @param model: Either a TabularDataModel or a ScrollableView, depending
        on what this column is part of.

        @param item: A port item instance (as defined by L{xmantissa.port}).

        @rtype: C{unicode}
        """
        certPath = super(CertificateColumn, self).extractValue(model, item)
        return certPath.path.decode('utf-8', 'replace')