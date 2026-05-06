def print_metadata(self, metadata, develop, active, installed_by):
        """
        Print out formatted metadata
        @param metadata: package's metadata
        @type metadata:  pkg_resources Distribution obj

        @param develop: path to pkg if its deployed in development mode
        @type develop: string

        @param active: show if package is activated or not
        @type active: boolean

        @param installed_by: Shows if pkg was installed by a package manager other
                             than setuptools
        @type installed_by: string

        @returns: None

        """
        show_metadata = self.options.metadata
        if self.options.fields:
            fields = self.options.fields.split(',')
            fields = map(str.strip, fields)
        else:
            fields = []
        version = metadata['Version']

        #When showing all packages, note which are not active:
        if active:
            if fields:
                active_status = ""
            else:
                active_status = "active"
        else:
            if fields:
                active_status = "*"
            else:
                active_status = "non-active"
        if develop:
            if fields:
                development_status = "! (%s)" % develop
            else:
                development_status = "development (%s)" % develop
        else:
            development_status = installed_by
        status = "%s %s" % (active_status, development_status)
        if fields:
            print('%s (%s)%s %s' % (metadata['Name'], version, active_status,
                                    development_status))
        else:
            # Need intelligent justification
            print(metadata['Name'].ljust(15) + " - " + version.ljust(12) + \
                " - " + status)
        if fields:
            #Only show specific fields, using case-insensitive search
            fields = map(str.lower, fields)
            for field in metadata.keys():
                if field.lower() in fields:
                    print('    %s: %s' % (field, metadata[field]))
            print()
        elif show_metadata:
            #Print all available metadata fields
            for field in metadata.keys():
                if field != 'Name' and field != 'Summary':
                    print('    %s: %s' % (field, metadata[field]))