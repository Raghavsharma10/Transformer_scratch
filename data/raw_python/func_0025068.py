def __cascade_delete_inner(self, master_item, safe: bool=False) -> typing.Optional[typing.Sequence]:
        """Cascade delete an item.

        Returns an undelete log that can be used to undo the cascade deletion.

        Builds a cascade of items to be deleted and dependencies to be removed when the passed item is deleted. Then
        removes computations that are no longer valid. Removing a computation may result in more deletions, so the
        process is repeated until nothing more gets removed.

        Next remove dependencies.

        Next remove individual items (from the most distant from the root item to the root item).
        """
        # print(f"cascade {master_item}")
        # this horrible little hack ensures that computation changed messages are delayed until the end of the cascade
        # delete; otherwise there are cases where dependencies can be reestablished during the changed messages while
        # this method is partially finished. ugh. see test_computation_deletes_when_source_cycle_deletes.
        if self.__computation_changed_delay_list is None:
            computation_changed_delay_list = list()
            self.__computation_changed_delay_list = computation_changed_delay_list
        else:
            computation_changed_delay_list = None
        undelete_log = list()
        try:
            items = list()
            dependencies = list()
            self.__build_cascade(master_item, items, dependencies)
            cascaded = True
            while cascaded:
                cascaded = False
                # adjust computation bookkeeping to remove deleted items, then delete unused computations
                items_set = set(items)
                for computation in copy.copy(self.computations):
                    output_deleted = master_item in computation._outputs
                    computation._inputs -= items_set
                    computation._outputs -= items_set
                    if computation not in items and computation != self.__current_computation:
                        # computations are auto deleted if all inputs are deleted or any output is deleted
                        if output_deleted or all(input in items for input in computation._inputs):
                            self.__build_cascade(computation, items, dependencies)
                            cascaded = True
            # print(list(reversed(items)))
            # print(list(reversed(dependencies)))
            for source, target in reversed(dependencies):
                self.__remove_dependency(source, target)
            # now delete the actual items
            for item in reversed(items):
                for computation in self.computations:
                    new_entries = computation.list_item_removed(item)
                    undelete_log.extend(new_entries)
                container = item.container
                if isinstance(item, DataItem.DataItem):
                    name = "data_items"
                elif isinstance(item, DisplayItem.DisplayItem):
                    name = "display_items"
                elif isinstance(item, Graphics.Graphic):
                    name = "graphics"
                elif isinstance(item, DataStructure.DataStructure):
                    name = "data_structures"
                elif isinstance(item, Symbolic.Computation):
                    name = "computations"
                elif isinstance(item, Connection.Connection):
                    name = "connections"
                elif isinstance(item, DisplayItem.DisplayDataChannel):
                    name = "display_data_channels"
                else:
                    name = None
                    assert False, "Unable to cascade delete type " + str(type(item))
                assert name
                # print(container, name, item)
                if container is self and name == "data_items":
                    # call the version of __remove_data_item that doesn't cascade again
                    index = getattr(container, name).index(item)
                    item_dict = item.write_to_dict()
                    # NOTE: __remove_data_item will notify_remove_item
                    undelete_log.extend(self.__remove_data_item(item, safe=safe))
                    undelete_log.append({"type": name, "index": index, "properties": item_dict})
                elif container is self and name == "display_items":
                    # call the version of __remove_data_item that doesn't cascade again
                    index = getattr(container, name).index(item)
                    item_dict = item.write_to_dict()
                    # NOTE: __remove_display_item will notify_remove_item
                    undelete_log.extend(self.__remove_display_item(item, safe=safe))
                    undelete_log.append({"type": name, "index": index, "properties": item_dict})
                elif container:
                    container_ref = str(container.uuid)
                    index = getattr(container, name).index(item)
                    item_dict = item.write_to_dict()
                    container_properties = container.save_properties() if hasattr(container, "save_properties") else dict()
                    undelete_log.append({"type": name, "container": container_ref, "index": index, "properties": item_dict, "container_properties": container_properties})
                    container.remove_item(name, item)
                    # handle top level 'remove item' notifications for data structures, computations, and display items here
                    # since they're not handled elsewhere.
                    if container == self and name in ("data_structures", "computations"):
                        self.notify_remove_item(name, item, index)
        except Exception as e:
            import sys, traceback
            traceback.print_exc()
            traceback.format_exception(*sys.exc_info())
        finally:
            # check whether this call of __cascade_delete is the top level one that will finish the computation
            # changed messages.
            if computation_changed_delay_list is not None:
                self.__finish_computation_changed()
        return undelete_log