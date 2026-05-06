def check_for_lane_permission(self):
        """
        One or more permissions can be associated with a lane
        of a workflow. In a similar way, a lane can be
        restricted with relation to other lanes of the workflow.

        This method called on lane changes and checks user has
        required permissions and relations.

        Raises:
             HTTPForbidden: if the current user hasn't got the
              required permissions and proper relations

        """
        # TODO: Cache lane_data in app memory
        if self.current.lane_permission:
            log.debug("HAS LANE PERM: %s" % self.current.lane_permission)
            perm = self.current.lane_permission
            if not self.current.has_permission(perm):
                raise HTTPError(403, "You don't have required lane permission: %s" % perm)

        if self.current.lane_relations:
            context = self.get_pool_context()
            log.debug("HAS LANE RELS: %s" % self.current.lane_relations)
            try:
                cond_result = eval(self.current.lane_relations, context)
            except:
                log.exception("CONDITION EVAL ERROR : %s || %s" % (
                    self.current.lane_relations, context))
                raise
            if not cond_result:
                log.debug("LANE RELATION ERR: %s %s" % (self.current.lane_relations, context))
                raise HTTPError(403, "You aren't qualified for this lane: %s" %
                                self.current.lane_relations)