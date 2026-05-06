def get_estimates_without_scope_in_month(self, customer):
        """
        It is expected that valid row for each month contains at least one
        price estimate for customer, service setting, service,
        service project link, project and resource.
        Otherwise all price estimates in the row should be deleted.
        """
        estimates = self.get_price_estimates_for_customer(customer)
        if not estimates:
            return []

        tables = {model: collections.defaultdict(list)
                  for model in self.get_estimated_models()}

        dates = set()
        for estimate in estimates:
            date = (estimate.year, estimate.month)
            dates.add(date)

            cls = estimate.content_type.model_class()
            for model, table in tables.items():
                if issubclass(cls, model):
                    table[date].append(estimate)
                    break

        invalid_estimates = []
        for date in dates:
            if any(map(lambda table: len(table[date]) == 0, tables.values())):
                for table in tables.values():
                    invalid_estimates.extend(table[date])
        print(invalid_estimates)
        return invalid_estimates