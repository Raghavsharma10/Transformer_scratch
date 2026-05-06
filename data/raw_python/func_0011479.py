def _handle_decl(self, node, scope, ctxt, stream):
        """TODO: Docstring for _handle_decl.

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling decl")

        metadata_processor = None
        if node.metadata is not None:
            #metadata_info = self._handle_metadata(node, scope, ctxt, stream)
            def process_metadata():
                metadata_info = self._handle_metadata(node, scope, ctxt, stream)
                return metadata_info
            metadata_processor = process_metadata

        field_name = self._get_node_name(node)
        field = self._handle_node(node.type, scope, ctxt, stream)
        bitsize = None
        bitfield_rw = None
        if getattr(node, "bitsize", None) is not None:
            bitsize = self._handle_node(node.bitsize, scope, ctxt, stream)
            has_prev = len(ctxt._pfp__children) > 0

            bitfield_rw = None
            if has_prev:
                prev = ctxt._pfp__children[-1]
                # if it was a bitfield as well
                # TODO I don't think this will handle multiple bitfield groups in a row.
                # E.g.
                #     char a: 8, b:8;
                #    char c: 8, d:8;
                if ((self._padded_bitfield and prev.__class__.width == field.width) or not self._padded_bitfield) \
                        and prev.bitsize is not None and prev.bitfield_rw.reserve_bits(bitsize, stream):
                    bitfield_rw = prev.bitfield_rw

            # either because there was no previous bitfield, or the previous was full
            if bitfield_rw is None:
                bitfield_rw = fields.BitfieldRW(self, field)
                bitfield_rw.reserve_bits(bitsize, stream)

        if getattr(node, "is_func_param", False):
            # we want to keep this as a class and not instantiate it
            # instantiation will be done in functions.ParamListDef.instantiate
            field = (field_name, field)
        
        # locals and consts still get a field instance, but DON'T parse the
        # stream!
        elif "local" in node.quals or "const" in node.quals:
            is_struct = issubclass(field, fields.Struct)
            if not isinstance(field, fields.Field) and not is_struct:
                field = field()
            scope.add_local(field_name, field)

            # this should only be able to be done with locals, right?
            # if not, move it to the bottom of the function
            if node.init is not None:
                val = self._handle_node(node.init, scope, ctxt, stream)
                if is_struct:
                    field = val
                    scope.add_local(field_name, field)
                else:
                    field._pfp__set_value(val)

            if "const" in node.quals:
                field._pfp__freeze()

            field._pfp__interp = self

        elif isinstance(field, functions.Function):
            # eh, just add it as a local...
            # maybe the whole local/vars thinking needs to change...
            # and we should only have ONE map TODO
            field.name = field_name
            scope.add_local(field_name, field)

        elif field_name is not None:
            added_child = False

    
            # by this point, structs are already instantiated (they need to be
            # in order to set the new context)
            if not isinstance(field, fields.Field):
                if issubclass(field, fields.NumberBase):
                    # use the default bitfield direction
                    if self._bitfield_direction is self.BITFIELD_DIR_DEFAULT:
                        bitfield_left_right = True if field.endian == fields.BIG_ENDIAN else False
                    else:
                        bitfield_left_right = (self._bitfield_direction is self.BITFIELD_DIR_LEFT_RIGHT)

                    field = field(
                        stream,
                        bitsize=bitsize,
                        metadata_processor=metadata_processor,
                        bitfield_rw=bitfield_rw,
                        bitfield_padded=self._padded_bitfield,
                        bitfield_left_right=bitfield_left_right
                    )

                # TODO
                # for now if there's a struct inside of a union that is being
                # parsed when there's an error, the user will lose information
                # about how far the parsing got. Here we are explicitly checking for
                # adding structs and unions to a parent union.
                elif (issubclass(field, fields.Struct) or issubclass(field, fields.Union)) \
                        and not isinstance(ctxt, fields.Union) \
                        and hasattr(field, "_pfp__init"):

                    # this is so that we can have all nested structs added to
                    # the root DOM, even if there's an error in parsing the data.
                    # If we didn't do this, any errors parsing the data would cause
                    # the new struct to not be added to its parent, and the user would
                    # not be able to see how far the script got
                    field = field(stream, metadata_processor=metadata_processor, do_init=False)
                    field._pfp__interp = self
                    field_res = ctxt._pfp__add_child(field_name, field, stream)

                    # when adding a new field to a struct/union/fileast, add it to the
                    # root of the ctxt's scope so that it doesn't get lost by being declared
                    # from within a function
                    scope.add_var(field_name, field_res, root=True)

                    field_res._pfp__interp = self
                    field._pfp__init(stream)
                    added_child = True
                else:
                    field = field(stream, metadata_processor=metadata_processor)

            if not added_child:
                field._pfp__interp = self
                field_res = ctxt._pfp__add_child(field_name, field, stream)
                field_res._pfp__interp = self

                # when adding a new field to a struct/union/fileast, add it to the
                # root of the ctxt's scope so that it doesn't get lost by being declared
                # from within a function
                scope.add_var(field_name, field_res, root=True)

                # this shouldn't be used elsewhere, but should still be explicit with
                # this flag
                added_child = True

        # enums will get here. If there is no name, then no
        # field is being declared (but the enum values _will_
        # get defined). E.g.:
        #     enum <uchar blah {
        #         BLAH1,
        #        BLAH2,
        #        BLAH3
        #     };
        elif field_name is None:
            pass

        return field