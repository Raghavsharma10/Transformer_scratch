def get_disease_comments(self, entry):
        """
        get list of models.Disease objects from XML node entry

        :param entry: XML node entry
        :return: list of :class:`pyuniprot.manager.models.Disease` objects
        """
        disease_comments = []
        query = "./comment[@type='disease']"

        for disease_comment in entry.iterfind(query):
            value_dict = {'comment': disease_comment.find('./text').text}

            disease = disease_comment.find("./disease")

            if disease is not None:
                disease_dict = {'identifier': disease.get('id')}

                for element in disease:
                    key = element.tag

                    if key in ['acronym', 'description', 'name']:
                        disease_dict[key] = element.text

                    if key == 'dbReference':
                        disease_dict['ref_id'] = element.get('id')
                        disease_dict['ref_type'] = element.get('type')

                disease_obj = models.get_or_create(self.session, models.Disease, **disease_dict)
                self.session.add(disease_obj)
                self.session.flush()
                value_dict['disease_id'] = disease_obj.id

            disease_comments.append(models.DiseaseComment(**value_dict))

        return disease_comments