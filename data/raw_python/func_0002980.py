def convert_def_list_elements(self):
        """
        A list in which each item consists of two parts: a word, phrase, term,
        graphic, chemical structure, or equation paired with one of more
        descriptions, discussions, explanations, or definitions of it.

        <def-list> elements are lists of <def-item> elements which are in turn
        composed of a pair of term (<term>) and definition (<def>). This method
        will convert the <def-list> to a classed <div> with a styled format
        for the terms and definitions.
        """
        for def_list in self.main.getroot().findall('.//def-list'):
            #Remove the attributes, excepting id
            remove_all_attributes(def_list, exclude=['id'])
            #Modify the def-list element
            def_list.tag = 'div'
            def_list.attrib['class'] = 'def-list'
            for def_item in def_list.findall('def-item'):
                #Get the term being defined, modify it
                term = def_item.find('term')
                term.tag = 'p'
                term.attrib['class']= 'def-item-term'
                #Insert it before its parent def_item
                insert_before(def_item, term)
                #Get the definition, handle missing with a warning
                definition = def_item.find('def')
                if definition is None:
                    log.warning('Missing def element in def-item')
                    remove(def_item)
                    continue
                #PLoS appears to consistently place all definition text in a
                #paragraph subelement of the def element
                def_para = definition.find('p')
                def_para.attrib['class'] = 'def-item-def'
                #Replace the def-item element with the p element
                replace(def_item, def_para)