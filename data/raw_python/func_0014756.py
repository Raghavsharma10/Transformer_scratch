def dashNameToCamelCase(dashName):
        '''
            dashNameToCamelCase - Converts a "dash name" (like padding-top) to its camel-case name ( like "paddingTop" )

            @param dashName <str> - A name containing dashes

                NOTE: This method is currently unused, but may be used in the future. kept for completeness.

            @return <str> - The camel-case form
        '''
        nameParts = dashName.split('-')
        for i in range(1, len(nameParts), 1):
            nameParts[i][0] = nameParts[i][0].upper()

        return ''.join(nameParts)