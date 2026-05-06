def _extract_body(self):
        """ Extract the body content from HTML. """

        def is_descendant_node(parent, node):
            node = node.getparent()
            while node is not None:
                if node == parent:
                    return True
                node = node.getparent()
            return False

        for pattern in self.config.body:
            items = self.parsed_tree.xpath(pattern)

            if len(items) == 1:
                if self.config.prune:
                    self.body = Document(etree.tostring(items[0])).summary()

                else:
                    self.body = etree.tostring(items[0])

                # We've got a body now.
                break

            else:
                appended_something = False
                body = etree.Element("root")

                for item in items:
                    if item.getparent() is None:
                        continue

                    is_descendant = False

                    for parent in body:
                        if (is_descendant_node(parent, item)):
                            is_descendant = True
                            break

                    if not is_descendant:

                        if self.config.prune:

                            # Clean with readability. Needs
                            # to-string conversion first.
                            pruned_string = Document(
                                etree.tostring(item)).summary()

                            # Re-parse the readability string
                            # output and include it in our body.
                            new_tree = etree.parse(
                                StringIO(pruned_string), self.parser)

                            failed = False

                            try:
                                body.append(
                                    new_tree.xpath('//html/body/div/div')[0]
                                )
                            except IndexError:

                                if 'id="readabilityBody"' in pruned_string:
                                    try:
                                        body.append(
                                            new_tree.xpath('//body')
                                        )
                                    except:
                                        failed = True

                                else:
                                    failed = True

                            if failed:
                                LOGGER.error(u'Pruning item failed:'
                                             u'\n\n%s\n\nWe got: “%s” '
                                             u'and skipped it.',
                                             etree.tostring(
                                                 item).replace(u'\n', u''),
                                             pruned_string.replace(u'\n', u''),
                                             extra={'siteconfig':
                                                    self.config.host})
                                pass

                        else:
                            body.append(item)

                        appended_something = True

                if appended_something:
                    self.body = etree.tostring(body)

                    # We've got a body now.
                    break