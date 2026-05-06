def transformer(self,elt_name,contents):
        """
        The OpenIOC indicator contains the actual observable bits of an indicator in the following
        form::

              <IndicatorItem id="b9ef2559-cc59-4463-81d9-52800545e16e" condition="contains">
                   <Context document="FileItem" search="FileItem/PEInfo/Sections/Section/Name" type="mir"/>
                   <Content type="string">.stub</Content>
              </IndicatorItem>

        We would rather have a key-value pairing of the following form (with the 'contains' attribute
        somewhere at the side::

             FileItem/PEInfo/Sections/Section/Name = .stub

        In order to achieve this, we create a DingoObjDict that corresponds to an XML that would
        look like follows::

             <FileItem id="b9ef2559-cc59-4463-81d9-52800545e16e">
              <PEInfo>
               <Sections>
                <Section>
                 <Name condition='contains' type='string'>
                  .stub
                 </Name>
                </Section>
               </Sections>
              </PEInfo>
             </FileItem>

        This is carried out by the transformer function, which is passed to the generic XML importer
        and executed for each element when converting the element into a dictionary structure.
        """

        # If the current element is not an IndicatorItem, we do nothing

        if elt_name != 'IndicatorItem':
            return (elt_name,contents)
        else:
            # We have an indicator item.

            # We initialize the resulting DingoObjDict
            result = DingoObjDict()

            # We initialize the dictionary that will contain the leaf (in the example given
            # above, that would be the dictionary representing the following bit of
            # XML::
            #
            #     <Name condition='contains'>
            #      .stub
            #     </Name>
            #

            leaf = DingoObjDict()

            # We extract the search term and split it into its elements (removing
            # the redundant first element '<something>Item'.)

            (document_type,search_term) = contents['Context']['@search'].split("/",1)
            search_term = search_term.split('/')

            # We extract the data for the leaf dictionary
            search_value = contents['Content']['_value']
            value_type = contents['Content']['@type']
            search_condition = contents['@condition']

            leaf['@value_type'] = value_type
            leaf['@condition'] = search_condition
            leaf['_value'] = search_value


            # We extract the identifier

            item_id = contents['@id']

            result['@id'] = item_id

            # We write the nested dictionary structure:

            set_dict(result,leaf,'set',*search_term)

        return (document_type,result)