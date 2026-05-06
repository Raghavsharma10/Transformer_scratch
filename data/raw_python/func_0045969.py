def get_group_id_to_child(self):
        """ At a minimum need a course composition parent and two children to the split test

            course composition
                    |
            split_test composition
                |           |
            vertical     vertical

        And the expected output is a URL-safe (&quot; instead of ") JSON string, of this object
            {
                0: "i4x://<org>/<course-name-slug>/<child-tag>/<child-name-slug>,
                1: "i4x://<org>/<course-name-slug>/<tag>/<child-name-slug>
            }
        """
        # get the children compositions, then construct
        # the escaped-JSON structure for this split_test
        group_ids = {}
        # also need the course name...so go up the composition tree
        course_node = None
        found_course = False
        rm = self.my_osid_object._get_provider_manager('REPOSITORY')
        if self.my_osid_object._proxy is not None:
            cqs = rm.get_composition_query_session_for_repository(Id(self.my_osid_object._my_map['assignedRepositoryIds'][0]),
                                                                  proxy=self.my_osid_object._proxy)
        else:
            cqs = rm.get_composition_query_session_for_repository(
                Id(self.my_osid_object._my_map['assignedRepositoryIds'][0]))
        search_node = self.my_osid_object
        while not found_course:
            querier = cqs.get_composition_query()
            cqs.use_unsequestered_composition_view()
            querier.match_contained_composition_id(search_node.ident, True)
            parents = cqs.get_compositions_by_query(querier)
            if parents.available() == 0:
                found_course = True
            else:
                parent = next(parents)
                if parent.genus_type.identifier == 'course':
                    found_course = True
                    course_node = parent
                else:
                    search_node = parent

        if course_node is None:
            return ''
        else:
            for index, child in enumerate(self.my_osid_object.get_children()):
                group_ids[index] = 'i4x://{0}/{1}/{2}/{3}'.format(course_node.org.text,
                                                                  re.sub('[^\w\s-]', '', course_node.display_name.text),
                                                                  child.genus_type.identifier,
                                                                  child.url)
            return json.dumps(group_ids).replace('"', '&quot;')