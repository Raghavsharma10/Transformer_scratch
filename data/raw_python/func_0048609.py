def get_edxml(self):
        """stub"""
        if self.has_raw_edxml():
            has_python = False
            my_files = self.my_osid_object.object_map['fileIds']
            raw_text = self.get_text('edxml').text
            soup = BeautifulSoup(raw_text, 'xml')
            # replace all file listings with an appropriate path...
            attrs = {
                'draggable': 'icon',
                'drag_and_drop_input': 'img',
                'files': 'included_files',
                'img': 'src'
            }
            local_regex = re.compile('[^http]')
            for key, attr in attrs.items():
                search = {attr: local_regex}
                tags = soup.find_all(**search)
                for item in tags:
                    if key == 'files' or item.name == key:
                        file_label = self._clean(item[attr])
                        if file_label in my_files:
                            content_type = Id(my_files[file_label]['assetContentTypeId'])
                            item[attr] = '/static/' + file_label + '.' + \
                                         content_type.get_identifier()

            # replace any python script with the item's get_text('python_script')
            # text...will fix weird whitespace issues
            if len(soup.find_all('script')) >= 1:
                scripts = soup.find_all('script')
                for script in scripts:
                    if 'python' in script['type']:
                        has_python = True
                        # contents = script.contents[0]
                        # contents.replaceWith(str(NavigableString(self.python)))
                        break

            try:
                if has_python:
                    return str(soup.find('problem'))
                else:
                    return soup.find('problem').prettify()
            except Exception:
                # if the edxml is not valid XML, it will not parse properly in soup
                # return just the raw edxml
                return self.get_text('edxml').text
        else:
            # have to construct the edxml from various components
            obj_map = self.my_osid_object.object_map
            question = obj_map['question']
            answers = obj_map['answers']
            if 'edx-multi-choice-problem-type' in obj_map['genusTypeId']:
                # get answer Ids to compare them to the choices
                answer_ids = []
                for answer in answers:
                    answer_ids += answer['choiceIds']
                # add the body text element (item.question.text)
                soup = BeautifulSoup('<problem></problem>', 'xml')
                p = soup.new_tag('p')
                p.string = self.get_text('questionString').text
                problem = soup.find('problem')
                problem.append(p)
                # add the metadata
                problem['display_name'] = question['displayName']['text']
                problem['showanswer'] = self.showanswer
                if 'rerandomize' in obj_map:
                    problem['rerandomize'] = obj_map['rerandomize']
                elif 'rerandomize' in question:
                    problem['rerandomize'] = question['rerandomize']
                problem['max_attempts'] = self.attempts

                # add the choices
                multichoice = soup.new_tag('multiplechoiceresponse')
                problem.append(multichoice)

                choicegroup = soup.new_tag('choicegroup')
                choicegroup['direction'] = 'vertical'
                multichoice.append(choicegroup)

                choices = question['choices']
                for choice in choices:
                    new_choice = soup.new_tag('choice')

                    # mark the correct choice(s)
                    if choice['id'] in answer_ids:
                        new_choice['correct'] = 'true'
                    else:
                        new_choice['correct'] = 'false'

                    new_choice['name'] = choice['name']
                    choice_text = soup.new_tag('text')
                    choice_text.string = choice['text']
                    new_choice.append(choice_text)
                    choicegroup.append(new_choice)
                return problem.prettify()
        raise IllegalState('records.assessment.edx.item_records.get_edxml()')