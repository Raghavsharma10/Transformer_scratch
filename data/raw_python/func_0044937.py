def constraints_stmt(stmt, env=None):
    """
    Since a statement may define new names or return an expression ,
    the constraints that result are in a
    ConstrainedEnv mapping names to types, with constraints, and maybe 
    having a return type (which is a constrained type)
    """
    env = env or {}
    
    if isinstance(stmt, ast.FunctionDef):
        arg_env = fn_env(stmt.args)

        body_env = extended_env(env, arg_env)
        constraints = []
        return_type = None # TODO: should be fresh and constrained?
        for body_stmt in stmt.body:
            cs = constraints_stmt(body_stmt, env=body_env)
            body_env.update(cs.env)
            constraints += cs.constraints
            return_type = union(return_type, cs.return_type)

        env[stmt.name] = Function(arg_types=[arg_env[arg.id] for arg in stmt.args.args],
                                  return_type=return_type)

        return ConstrainedEnv(env=env, constraints=constraints)

    elif isinstance(stmt, ast.Expr):
        constrained_ty = constraints_expr(stmt.value, env=env)
        return ConstrainedEnv(env=env, constraints=constrained_ty.constraints)
        
    elif isinstance(stmt, ast.Return):
        if stmt.value:
            expr_result = constraints_expr(stmt.value, env=env)
            return ConstrainedEnv(env=env, constraints=expr_result.constraints, return_type=expr_result.type)
        else:
            result = fresh()
            return ConstrainedEnv(env=env, constraints=[Constraint(subtype=result, supertype=NamedType('NoneType'))])

    elif isinstance(stmt, ast.Assign):
        if len(stmt.targets) > 1:
            raise NotImplementedError('Cannot generate constraints for multi-target assignments yet')

        expr_result = constraints_expr(stmt.value, env=env)
        target = stmt.targets[0].id
        
        # For an assignment, we actually generate a fresh variable so that it can be the union of all things assigned
        # to it. We do not do any typestate funkiness.
        if target not in env:
            env[target] = fresh()
            
        return ConstrainedEnv(env=env, 
                              constraints = expr_result.constraints + [Constraint(subtype=expr_result.type, 
                                                                                  supertype=env[target])])

    else:
        raise NotImplementedError('Constraint gen for stmt %s' % stmt)