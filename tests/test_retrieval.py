from backend.rag.retriever import shortest_path, _fk_graph


def test_average_rating_of_vehicles(ctx):
    r = ctx.retriever.retrieve("average rating of vehicles")
    assert {"vehicles", "rental", "rating"} <= set(r.tables)


def test_customers_from_delhi(ctx):
    r = ctx.retriever.retrieve("customers from Delhi")
    assert r.ranked[0][0] == "customers"


def test_maintenance_question(ctx):
    assert ctx.retriever.retrieve("total maintenance cost per vehicle").ranked[0][0] == "maintenance"


def test_relationship_expansion_bridges_rating_to_vehicles(ctx):
    r = ctx.retriever.retrieve("rating of each vehicle", top_k=2, expand=True)
    assert "rental" in r.tables  # not in the top-2 vector hits, added as bridge


def test_shortest_path(schema):
    assert shortest_path(_fk_graph(schema), "rating", "vehicles") == ["rating", "rental", "vehicles"]


def test_top_k_configurable(ctx):
    assert len(ctx.retriever.retrieve("customers", top_k=2, expand=False).tables) == 2


def test_schema_text_lists_join_conditions(ctx):
    assert "rental.vehicle_id = vehicles.vehicle_id" in ctx.retriever.retrieve("rating vehicle").schema_text
