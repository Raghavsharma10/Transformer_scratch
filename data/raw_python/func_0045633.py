def __to_plain_containers(self,
                              container: Union[CommentedSeq, CommentedMap]
                              ) -> Union[OrderedDict, list]:
        """Converts any sequence or mapping to list or OrderedDict

        Stops at anything that isn't a sequence or a mapping.

        One day, we'll extract the comments and formatting and store \
        them out-of-band.

        Args:
            mapping: The mapping of constructed subobjects to edit
        """
        if isinstance(container, CommentedMap):
            new_container = OrderedDict()  # type: Union[OrderedDict, list]
            for key, value_obj in container.items():
                if (isinstance(value_obj, CommentedMap)
                        or isinstance(value_obj, CommentedSeq)):
                    new_container[key] = self.__to_plain_containers(value_obj)
                else:
                    new_container[key] = value_obj

        elif isinstance(container, CommentedSeq):
            new_container = list()
            for value_obj in container:
                if (isinstance(value_obj, CommentedMap)
                        or isinstance(value_obj, CommentedSeq)):
                    new_container.append(self.__to_plain_containers(value_obj))
                else:
                    new_container.append(value_obj)
        return new_container