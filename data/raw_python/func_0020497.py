def update_annotations_on_build(self, build_id, annotations):
        """
        set annotations on build object

        :param build_id: str, id of build
        :param annotations: dict, annotations to set
        :return:
        """
        return self.adjust_attributes_on_object('builds', build_id,
                                                'annotations', annotations,
                                                self._update_metadata_things)