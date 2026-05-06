def recursive_article_navmap(self, src_element, depth=0, first=True):
        """
        This function recursively traverses the content of an input article to
        add the correct elements to the NCX file's navMap and Lists.
        """
        if depth > self.nav_depth:
            self.nav_depth = depth
        navpoints = []
        tagnames = ['sec', 'fig', 'table-wrap']
        for child in src_element:
            try:
                tagname = child.tag
            except AttributeError:
                continue
            else:
                if tagname not in tagnames:
                    continue

            #Safely handle missing id attributes
            if 'id' not in child.attrib:
                child.attrib['id'] = self.auto_id

            #If in collection mode, we'll prepend the article DOI to avoid
            #collisions
            if self.collection:
                child_id = '-'.join([self.article_doi,
                                     child.attrib['id']])
            else:
                child_id = child.attrib['id']

            #Attempt to infer the correct text as a label
            #Skip the element if we cannot
            child_title = child.find('title')
            if child_title is None:
                continue  # If there is no immediate title, skip this element
            label = element_methods.all_text(child_title)
            if not label:
                continue  # If no text in the title, skip this element
            source = 'main.{0}.xhtml#{1}'.format(self.article_doi,
                                               child.attrib['id'])
            if tagname == 'sec':
                children = self.recursive_article_navmap(child, depth=depth + 1)
                navpoints.append(navpoint(child_id,
                                          label,
                                          self.play_order,
                                          source,
                                          children))
            #figs and table-wraps do not have children
            elif tagname == 'fig':  # Add navpoints to list_of_figures
                self.figures_list.append(navpoint(child.attrib['id'],
                                                  label,
                                                  None,
                                                  source,
                                                  []))
            elif tagname == 'table-wrap':  # Add navpoints to list_of_tables
                self.tables_list.append(navpoint(child.attrib['id'],
                                                 label,
                                                 None,
                                                 source,
                                                 []))
        return navpoints