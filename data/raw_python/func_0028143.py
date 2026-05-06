def clean_dataset_tags(self):
        # type: () -> Tuple[bool, bool]
        """Clean dataset tags according to tags cleanup spreadsheet and return if any changes occurred

        Returns:
            Tuple[bool, bool]: Returns (True if tags changed or False if not, True if error or False if not)
        """
        tags_dict, wildcard_tags = Tags.tagscleanupdicts()

        def delete_tag(tag):
            logger.info('%s - Deleting tag %s!' % (self.data['name'], tag))
            return self.remove_tag(tag), False

        def update_tag(tag, final_tags, wording, remove_existing=True):
            text = '%s - %s: %s -> ' % (self.data['name'], wording, tag)
            if not final_tags:
                logger.error('%snothing!' % text)
                return False, True
            tags_lower_five = final_tags[:5].lower()
            if tags_lower_five == 'merge' or tags_lower_five == 'split' or (
                    ';' not in final_tags and len(final_tags) > 50):
                logger.error('%s%s - Invalid final tag!' % (text, final_tags))
                return False, True
            if remove_existing:
                self.remove_tag(tag)
            tags = ', '.join(self.get_tags())
            if self.add_tags(final_tags.split(';')):
                logger.info('%s%s! Dataset tags: %s' % (text, final_tags, tags))
            else:
                logger.warning(
                    '%s%s - At least one of the tags already exists! Dataset tags: %s' % (text, final_tags, tags))
            return True, False

        def do_action(tag, tags_dict_key):
            whattodo = tags_dict[tags_dict_key]
            action = whattodo[u'action']
            final_tags = whattodo[u'final tags (semicolon separated)']
            if action == u'Delete':
                changed, error = delete_tag(tag)
            elif action == u'Merge':
                changed, error = update_tag(tag, final_tags, 'Merging')
            elif action == u'Fix spelling':
                changed, error = update_tag(tag, final_tags, 'Fixing spelling')
            elif action == u'Non English':
                changed, error = update_tag(tag, final_tags, 'Anglicising', remove_existing=False)
            else:
                changed = False
                error = False
            return changed, error

        def process_tag(tag):
            changed = False
            error = False
            if tag in tags_dict.keys():
                changed, error = do_action(tag, tag)
            else:
                for wildcard_tag in wildcard_tags:
                    if fnmatch.fnmatch(tag, wildcard_tag):
                        changed, error = do_action(tag, wildcard_tag)
                        break
            return changed, error

        anychange = False
        anyerror = False
        for tag in self.get_tags():
            changed, error = process_tag(tag)
            if changed:
                anychange = True
            if error:
                anyerror = True

        return anychange, anyerror