def receive(self, data, api_context):
        """Pass an API result down the pipeline"""
        self.log.debug(f"Putting data on the pipeline: {data}")
        result = {
            "api_contexts": self.api_contexts,
            "api_context": api_context,
            "strategy": dict(),  # Shared strategy data
            "result": data,
            "log_level": api_context["log_level"],
        }
        self.strat.execute(self.strategy_context_schema().load(result).data)