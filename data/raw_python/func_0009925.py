def DOMDebugger_setDOMBreakpoint(self, nodeId, type):
		"""
		Function path: DOMDebugger.setDOMBreakpoint
			Domain: DOMDebugger
			Method name: setDOMBreakpoint
		
			Parameters:
				Required arguments:
					'nodeId' (type: DOM.NodeId) -> Identifier of the node to set breakpoint on.
					'type' (type: DOMBreakpointType) -> Type of the operation to stop upon.
			No return value.
		
			Description: Sets breakpoint on particular operation with DOM.
		"""
		subdom_funcs = self.synchronous_command('DOMDebugger.setDOMBreakpoint',
		    nodeId=nodeId, type=type)
		return subdom_funcs