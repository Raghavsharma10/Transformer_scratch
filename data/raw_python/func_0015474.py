def done_evaluating(self, n: Node, s: ShExJ.shapeExpr, result: bool) -> Tuple[bool, bool]:
        """
        Indicate that we have completed an actual evaluation of (n,s).  This is only called when start_evaluating
        has returned None as the assumed result

        :param n: Node that was evaluated
        :param s: expression for node evaluation
        :param result: result of evaluation
        :return: Tuple - first element is whether we are done, second is whether evaluation was consistent
        """
        key = (n, s.id)

        # If we didn't have to assume anything or our assumption was correct, we're done
        if key not in self.assumptions or self.assumptions[key] == result:
            if key in self.assumptions:
                del self.assumptions[key]       # good housekeeping, not strictly necessary
            self.evaluating.remove(key)
            self.known_results[key] = result
            return True, True
        # If we assumed true and got a false, try assuming false
        elif self.assumptions[key]:
            self.evaluating.remove(key)         # restart the evaluation from the top
            self.assumptions[key] = False
            return False, True
        else:
            self.fail_reason = f"{s.id}: Inconsistent recursive shape reference"
            return True, False