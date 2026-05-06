def _filtering_result_checked(self, by_or):
		'''Check if post passes all / at_least_one (by_or parameter) filter(s).
			Filters are evaluated on only-if-necessary ("lazy") basis.'''
		filters, results = it.imap(set, ( self.feed.filters.all(),
			self.filtering_results.values_list('filter', flat=True) ))

		# Check if conclusion can already be made, based on cached results.
		if results.issubset(filters):
			# If at least one failed/passed test is already there, and/or outcome is defined.
			try: return self._filtering_result(by_or)
			except IndexError: # inconclusive until results are consistent
				if filters == results: return not by_or

		# Consistency check / update.
		if filters != results:
			# Drop obsolete (removed, unbound from feed)
			#  filters' results (they WILL corrupt outcome).
			self.filtering_results.filter(filter__in=results.difference(filters)).delete()
			# One more try, now that results are only from feed filters' subset.
			try: return self._filtering_result(by_or)
			except IndexError: pass
			# Check if any filter-results are not cached yet, create them (perform actual filtering).
			# Note that independent filters applied first, since
			#  crossrefs should be more resource-hungry in general.
			for filter_obj in sorted(filters.difference(results), key=op.attrgetter('base.crossref')):
				filter_op = FilterResult(filter=filter_obj, post=self, result=filter_obj.handler(self))
				filter_op.save()
				if filter_op.result == by_or: return by_or # return as soon as first passed / failed

		# Final result
		try: return self._filtering_result(by_or)
		except IndexError: return not by_or