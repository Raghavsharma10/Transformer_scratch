def DOMDebugger_removeDOMBreakpoint(self, nodeId, type):
		"""
		Function path: DOMDebugger.removeDOMBreakpoint
			Domain: DOMDebugger
			Method name: removeDOMBreakpoint
		
			Parameters:
				Required arguments:
					'nodeId' (type: DOM.NodeId) -> Identifier of the node to remove breakpoint from.
					'type' (type: DOMBreakpointType) -> Type of the breakpoint to remove.
			No return value.
		
			Description: Removes DOM breakpoint that was set using <code>setDOMBreakpoint</code>.
		"""
		subdom_funcs = self.synchronous_command('DOMDebugger.removeDOMBreakpoint',
		    nodeId=nodeId, type=type)
		return subdom_funcs