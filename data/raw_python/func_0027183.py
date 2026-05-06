def compiler_dialect(paramstyle='named'):
    """
    构建dialect
    """
    dialect = SQLiteDialect_pysqlite(
        json_serializer=json.dumps,
        json_deserializer=json_deserializer,
        paramstyle=paramstyle
    )
    dialect.default_paramstyle = paramstyle
    dialect.statement_compiler = ACompiler_sqlite
    return dialect