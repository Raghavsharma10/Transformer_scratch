def _update_new_linked_model(self, internal, linked_mdl_ins, link):
        """
        Iterates through linked_models of given model instance to match it's
        "reverse" with given link's "field" values.
        """

        # If there is a link between two sides (A and B), if a link from A to B,
        # link should be saved at B but it is not necessary to control again data in A.
        # If internal field is True, data control is not done and passes.
        if not internal:

            for lnk in linked_mdl_ins.get_links():
                mdl = lnk['mdl']
                if not isinstance(self, mdl) or lnk['reverse'] != link['field']:
                    continue
                local_field_name = lnk['field']
                # remote_name = lnk['reverse']
                remote_field_name = un_camel(mdl.__name__)
                if not link['o2o']:
                    if '.' in local_field_name:
                        local_field_name, remote_field_name = local_field_name.split('.')
                    remote_set = getattr(linked_mdl_ins, local_field_name)

                    if remote_set._TYPE == 'ListNode' and self not in remote_set:
                        remote_set(**{remote_field_name: self._root_node})
                        if linked_mdl_ins._exists is False:
                            raise ObjectDoesNotExist('Linked %s on field %s with key %s doesn\'t exist' % (
                                linked_mdl_ins.__class__.__name__,
                                remote_field_name,
                                linked_mdl_ins.key,
                            ))
                        linked_mdl_ins.save(internal=True)
                else:
                    linked_mdl_ins.setattr(remote_field_name, self._root_node)
                    if linked_mdl_ins._exists is False:
                        raise ObjectDoesNotExist('Linked object %s on field %s with key %s doesn\'t exist' % (
                            linked_mdl_ins.__class__.__name__,
                            remote_field_name,
                            linked_mdl_ins.key,
                        ))
                    linked_mdl_ins.save(internal=True)