def openioc_embedding_pred(self,parent, child, ns_mapping):
        """
        Predicate for recognizing inlined content in an XML; to
        be used for DINGO's xml-import hook 'embedded_predicate'.
        The question this predicate must answer is whether
        the child should be extracted into a separate object.

        The function returns either
        - False (the child is not to be extracted)
        - True (the child is extracted but nothing can be inferred
          about what kind of object is extracted)
        - a string giving some indication about the object type
          (if nothing else is known: the name of the element, often the
           namespace of the embedded object)
        - a dictionary, of the following form::

             {'id_and_revision_info' : { 'id': something/None,
                                         'ts': something/None,
                                          ... other information you want to
                                          record for this object for later usage,
                                       },
              'embedded_ns': False/True/some indication about object type as string}

        Note: the 'parent' and 'child' arguments are XMLNodes as defined
        by the Python libxml2 bindings. If you have never worked with these, have a look at

        - Mike Kneller's brief intro: http://mikekneller.com/kb/python/libxml2python/part1
        - the functions in django-dingos core.xml_utils module
        """

        # For openIOC, we extract the Indicator-Item elements,
        # since those correspond to observables.

        child_attributes = extract_attributes(child,prefix_key_char='')


        if ('id' in child_attributes and child.name == 'IndicatorItem'):

            # The embedding predicate is supposed to not only return
            # 'True' or 'False', but in case there is an embedding,
            # it should also contain information regarding the type of
            # object that is embedded. This is used, for example, to
            # create the DataType  information for the embedding element
            # (it is a reference to an object of type X).

            # In OpenIOC, The IndicatorItems have the following form::
            #
            #      <IndicatorItem id="b9ef2559-cc59-4463-81d9-52800545e16e" condition="contains">
            #          <Context document="FileItem" search="FileItem/PEInfo/Sections/Section/Name" type="mir"/>
            #          <Content type="string">.stub</Content>
            #      </IndicatorItem>
            #
            # We take the 'document' attribute of the 'Context' element as object type
            # of the embedded object (as we shall see below, upon import, we rewrite
            # the IndicatorItem such that it corresponds to the 'fact_term = value' structure
            # used for STIX/CybOX data.

            grandchild = child.children
            type_info = None

            while grandchild is not None:
                if grandchild.name == 'Context':
                    context_attributes = extract_attributes(grandchild,prefix_key_char='')
                    if 'document' in context_attributes:
                        type_info = context_attributes['document']
                    break
                grandchild = grandchild.next

            if type_info:
                return type_info
            else:
                return True
        else:
            return False