def constructSpec(indentation, begin_block, end_block, begin_line, end_line, 
                  begin_action, end_action, 
                  begin_condition, end_condition, 
                  logical_and, logical_or):
    """Return a language specification based on parameters."""
    return {
        INDENTATION   : indentation, 
        BEG_BLOCK     : begin_block,
        END_BLOCK     : end_block,
        BEG_LINE      : begin_line, 
        END_LINE      : end_line, 
        BEG_ACTION    : begin_action, 
        END_ACTION    : end_action, 
        BEG_CONDITION : begin_condition, 
        END_CONDITION : end_condition, 
        LOGICAL_AND   : logical_and, 
        LOGICAL_OR    : logical_or
    }