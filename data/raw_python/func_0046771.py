def initialize(self, magic_identifier, assessment_section):
        """This method is to be called by a magic AssessmentPart lookup session.

        magic_identifier_part includes:
            parent_id = id string of the parent part that created this part
            level = how many levels deep
            objective_id = the Objective Id to for which to select an item
            waypoint_index = the index of this item in its parent part

        """
        arg_map = json.loads(unquote(magic_identifier).split('?')[-1],
                             object_pairs_hook=OrderedDict)
        self._magic_identifier = magic_identifier
        self._assessment_section = assessment_section
        if 'level' in arg_map:
            self._level = arg_map['level']
        else:
            self._level = 0
        if 'parent_id' in arg_map:
            self._magic_parent_id = Id(arg_map['parent_id'])
        self.my_osid_object._my_map['learningObjectiveIds'] = arg_map['objective_ids']
        self.my_osid_object._my_map['waypointIndex'] = arg_map['waypoint_index']

        if self.my_osid_object._my_map['learningObjectiveIds'] != ['']:
            try:
                self.my_osid_object._my_map['itemIds'] = [str(self.get_my_item_id_from_section(assessment_section))]
            except IllegalState:
                self.load_item_for_objective()
            except AttributeError:
                # when the magic part is being retrieved without a section ...
                # i.e. when authoring, but no itemId explicitly set (perhaps it
                #      was only set with a learningObjectiveId)
                self.my_osid_object._my_map['itemIds'] = []