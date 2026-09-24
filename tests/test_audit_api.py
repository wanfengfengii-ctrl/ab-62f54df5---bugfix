"""End-to-end audit behavior through the versioned HTTP API."""

import math

import pytest


def post(client, body, path="/v1/audit"):
    resp = client.post(path, json=body)
    return resp.status_code, resp.json()


# ----------------------------------------------------------------- health
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert client.get("/v1/health").json()["status"] == "ok"


def test_version_index(client):
    body = client.get("/v1").json()
    assert body["api"] == "v1"


# --------------------------------------------------------------- validation
def test_json_numbers_rejected(client):
    code, body = post(client, {
        "contour": [
            {"type": "line", "start": [0, 0], "end": [1, 1]},
        ],
        "tool_radius": "1",
    })
    assert code == 422
    assert body["status"] == "rejected"
    assert body["error"]["code"] == "invalid_request"


def test_non_decimal_text_rejected(client):
    code, body = post(client, {
        "contour": [
            {"type": "line", "start": ["0", "0"], "end": ["1.0", "0"]},
        ],
        "tool_radius": "1/2",
    })
    assert code == 422
    assert body["error"]["code"] == "invalid_request"


def test_chain_gap_rejected(client):
    code, body = post(client, {
        "contour": [
            {"type": "line", "start": ["0", "0"], "end": ["1", "0"]},
            {"type": "line", "start": ["2", "0"], "end": ["2", "1"]},
        ],
        "tool_radius": "0.1",
    })
    assert code == 422
    assert body["error"]["code"] == "chain_not_closed"


# ------------------------------------------------------------------ square
def test_square_inner_compensation(client):
    code, body = post(client, {"contour": [
        {"type": "line", "start": ["0", "0"], "end": ["10", "0"]},
        {"type": "line", "start": ["10", "0"], "end": ["10", "10"]},
        {"type": "line", "start": ["10", "10"], "end": ["0", "10"]},
        {"type": "line", "start": ["0", "10"], "end": ["0", "0"]},
    ], "tool_radius": "1", "side": "left"})
    assert code == 200, body
    cc = body["compensated_contour"]
    assert cc["orientation"] == "ccw"
    assert cc["turning_number"] == 1
    assert len(cc["pieces"]) == 4
    lines = cc["pieces"]
    assert all(p["type"] == "line" for p in lines)
    assert float(cc["signed_area"]["decimal"]) == pytest.approx(64.0, abs=1e-9)
    # Exact rational coordinates, no radicals needed for an inner square.
    first = lines[0]["start"]
    assert first["x"]["exact"]["rational"] == [1, 1]
    assert first["y"]["exact"]["rational"] == [1, 1]


def test_square_outer_compensation_four_fillets(client):
    code, body = post(client, {"contour": [
        {"type": "line", "start": ["0", "0"], "end": ["10", "0"]},
        {"type": "line", "start": ["10", "0"], "end": ["10", "10"]},
        {"type": "line", "start": ["10", "10"], "end": ["0", "10"]},
        {"type": "line", "start": ["0", "10"], "end": ["0", "0"]},
    ], "tool_radius": "1", "side": "right"})
    assert code == 200, body
    cc = body["compensated_contour"]
    assert len(cc["pieces"]) == 8
    fillets = [p for p in cc["pieces"] if p["type"] == "join_fillet"]
    assert len(fillets) == 4
    assert all(f["radius"]["exact"]["rational"] == [1, 1] for f in fillets)
    # Area = (10+2)^2 - 4 + pi = 140 + pi.
    assert float(cc["signed_area"]["decimal"]) == pytest.approx(
        140 + math.pi, abs=1e-9
    )


# --------------------------------------------------------------- full circle
def test_circle_inner_offset_and_collapse(client):
    code, body = post(client, {"contour": [
        {"type": "arc", "start": ["10", "0"], "end": ["10", "0"],
         "center": ["0", "0"], "sense": "ccw", "full": True},
    ], "tool_radius": "1", "side": "left"})
    assert code == 200, body
    cc = body["compensated_contour"]
    assert cc["pieces"][0]["radius"]["decimal"].startswith("9.")
    assert float(cc["signed_area"]["decimal"]) == pytest.approx(
        81 * math.pi, abs=1e-8
    )

    code, body = post(client, {"contour": [
        {"type": "arc", "start": ["1", "0"], "end": ["1", "0"],
         "center": ["0", "0"], "sense": "ccw", "full": True},
    ], "tool_radius": "1", "side": "left"})
    assert code == 409
    assert body["status"] == "unsafe"
    assert body["error"]["code"] == "radius_collapse"


def test_circle_inversion_rejected(client):
    code, body = post(client, {"contour": [
        {"type": "arc", "start": ["0.5", "0"], "end": ["0.5", "0"],
         "center": ["0", "0"], "sense": "ccw", "full": True},
    ], "tool_radius": "1", "side": "left"})
    assert code == 409
    assert body["error"]["code"] == "radius_inversion"
    # The defect must be traceable to its source entity.
    assert body["error"]["first_defect"]["source_entities"][0]["index"] == 0


# -------------------------------------------------------------- D contours
def test_d_inner_line_circle_concave_join(client):
    code, body = post(client, {
        "contour": [
            {"type": "arc", "start": ["5", "0"], "end": ["-5", "0"],
             "center": ["0", "0"], "sense": "ccw"},
            {"type": "line", "start": ["-5", "0"], "end": ["5", "0"]},
        ],
        "tool_radius": "1", "side": "left",
    })
    assert code == 200, body
    cc = body["compensated_contour"]
    assert len(cc["pieces"]) == 2  # sharp concave joins, no fillets
    line = next(p for p in cc["pieces"] if p["type"] == "line")
    assert float(line["start"]["y"]["decimal"]) == pytest.approx(1.0)
    # The line runs -sqrt(15) -> +sqrt(15).
    assert abs(abs(float(line["start"]["x"]["decimal"])) - math.sqrt(15)) < 1e-9
    assert abs(float(line["end"]["x"]["decimal"]) - math.sqrt(15)) < 1e-9
    # Independent numeric area: chord term -sqrt(15), arc sector 8*(pi-2a).
    alpha = math.asin(0.25)
    expected = -math.sqrt(15) + 8 * (math.pi - 2 * alpha)
    assert float(cc["signed_area"]["decimal"]) == pytest.approx(expected, abs=1e-8)


def test_all_four_d_orientations(client):
    def body_for(sense, side):
        if sense == "ccw":
            contour = [
                {"type": "arc", "start": ["5", "0"], "end": ["-5", "0"],
                 "center": ["0", "0"], "sense": "ccw"},
                {"type": "line", "start": ["-5", "0"], "end": ["5", "0"]},
            ]
        else:
            contour = [
                {"type": "line", "start": ["5", "0"], "end": ["-5", "0"]},
                {"type": "arc", "start": ["-5", "0"], "end": ["5", "0"],
                 "center": ["0", "0"], "sense": "cw"},
            ]
        return post(client, {"contour": contour,
                             "tool_radius": "1", "side": side})

    expectations = {
        ("ccw", "left"): (1, 17.216873800238),
        ("ccw", "right"): (1, 68.119464091411),
        ("cw", "left"): (-1, -68.119464091411),
        ("cw", "right"): (-1, -17.216873800238),
    }
    for (sense, side), (tn, area) in expectations.items():
        code, body = body_for(sense, side)
        assert code == 200, (sense, side, body)
        cc = body["compensated_contour"]
        assert cc["turning_number"] == tn
        assert float(cc["signed_area"]["decimal"]) == pytest.approx(area, abs=1e-8)


def test_smooth_capsule_has_no_fillets(client):
    code, body = post(client, {
        "contour": [
            {"type": "arc", "start": ["3", "-1"], "end": ["3", "1"],
             "center": ["3", "0"], "sense": "ccw"},
            {"type": "line", "start": ["3", "1"], "end": ["-3", "1"]},
            {"type": "arc", "start": ["-3", "1"], "end": ["-3", "-1"],
             "center": ["-3", "0"], "sense": "ccw"},
            {"type": "line", "start": ["-3", "-1"], "end": ["3", "-1"]},
        ],
        "tool_radius": "0.5", "side": "right",
    })
    assert code == 200, body
    cc = body["compensated_contour"]
    assert len(cc["pieces"]) == 4
    assert all(p["type"] != "join_fillet" for p in cc["pieces"])


# ------------------------------------------------------------- simplicity
def test_self_intersecting_source_reported(client):
    bow = [
        {"type": "line", "start": ["0", "0"], "end": ["10", "10"]},
        {"type": "line", "start": ["10", "10"], "end": ["0", "10"]},
        {"type": "line", "start": ["0", "10"], "end": ["10", "0"]},
        {"type": "line", "start": ["10", "0"], "end": ["0", "0"]},
    ]
    code, body = post(client, {"contour": bow, "tool_radius": "0.5",
                               "side": "left"})
    assert code == 409
    err = body["error"]
    assert err["code"] == "self_intersection"
    fd = err["first_defect"]
    assert "tool_path_position" in fd
    assert "exact_point" in fd
    assert {e["index"] for e in fd["source_entities"]} == {0, 2}


# The reported closed tool path has an early self-intersection (entities 0
# and 2) and a separate later contact in its tail.  The early defect must be
# reported at its exact point; the tail contact must not mask it or turn the
# request into a 500.
_REPORTED_CONTOUR = [
    {"type": "line", "start": ["0", "0"], "end": ["3", "1"]},
    {"type": "line", "start": ["3", "1"], "end": ["1", "3"]},
    {"type": "line", "start": ["1", "3"], "end": ["0", "0"]},
    {"type": "line", "start": ["0", "0"], "end": ["-1", "3"]},
    {"type": "line", "start": ["-1", "3"], "end": ["2", "-4"]},
    {"type": "line", "start": ["2", "-4"], "end": ["0", "0"]},
]

# Same early bow, tail rerouted well clear of the early pieces.
_TAIL_REMOVED_CONTROL = [
    {"type": "line", "start": ["0", "0"], "end": ["3", "1"]},
    {"type": "line", "start": ["3", "1"], "end": ["1", "3"]},
    {"type": "line", "start": ["1", "3"], "end": ["0", "0"]},
    {"type": "line", "start": ["0", "0"], "end": ["-4", "0"]},
    {"type": "line", "start": ["-4", "0"], "end": ["-4", "-4"]},
    {"type": "line", "start": ["-4", "-4"], "end": ["0", "0"]},
]


def test_reported_contour_early_self_intersection(client):
    code, body = post(client, {"contour": _REPORTED_CONTOUR,
                               "tool_radius": "0.1", "side": "left"})
    assert code == 409, body
    assert body["status"] == "unsafe"
    fd = body["error"]["first_defect"]
    assert body["error"]["code"] == "self_intersection"
    # Earliest defect is on piece 0 (sourced from entities 0 and 2).
    assert fd["tool_path_position"]["piece_index"] == 0
    assert fd["tool_path_position"]["intra"] == "interior"
    assert [e["index"] for e in fd["source_entities"]] == [0, 2]
    # Exact, recomputable point: (sqrt(10)/20, sqrt(10)/20).
    expected = {"*": [{"rational": [1, 20]},
                      {"sqrt": {"rational": [10, 1]}}]}
    assert fd["exact_point"]["x"]["exact"] == expected
    assert fd["exact_point"]["y"]["exact"] == expected
    assert math.isclose(
        float(fd["exact_point"]["x"]["decimal"]),
        math.sqrt(10) / 20, rel_tol=1e-9,
    )


def test_early_defect_unchanged_without_tail_contact(client):
    code_full, body_full = post(
        client, {"contour": _REPORTED_CONTOUR,
                 "tool_radius": "0.1", "side": "left"})
    code_ctrl, body_ctrl = post(
        client, {"contour": _TAIL_REMOVED_CONTROL,
                 "tool_radius": "0.1", "side": "left"})
    assert code_full == code_ctrl == 409
    f, g = (body_full["error"]["first_defect"],
            body_ctrl["error"]["first_defect"])
    assert body_ctrl["error"]["code"] == "self_intersection"
    assert f["tool_path_position"] == g["tool_path_position"]
    assert f["source_entities"] == g["source_entities"]
    assert f["exact_point"]["x"]["exact"] == g["exact_point"]["x"]["exact"]
    assert f["exact_point"]["y"]["exact"] == g["exact_point"]["y"]["exact"]



def test_oversized_tool_narrow_slot_rejected(client):
    code, body = post(client, {"contour": [
        {"type": "line", "start": ["0", "0"], "end": ["3", "0"]},
        {"type": "line", "start": ["3", "0"], "end": ["3", "10"]},
        {"type": "line", "start": ["3", "10"], "end": ["0", "10"]},
        {"type": "line", "start": ["0", "10"], "end": ["0", "0"]},
    ], "tool_radius": "2", "side": "left"})
    assert code == 409
    assert body["status"] == "unsafe"
    assert body["error"]["code"] == "trim_order_reversal"
    fd = body["error"]["first_defect"]
    # Exact, recomputable trim points are returned.
    assert fd["exact_point"]["x"]["exact"]["rational"] == [2, 1]


def test_tangent_nonadjacent_pinch_is_defect(client):
    # Two lobes meeting at one shared point: compensated chain pinches.
    code, body = post(client, {"contour": [
        {"type": "line", "start": ["0", "0"], "end": ["5", "0"]},
        {"type": "line", "start": ["5", "0"], "end": ["5", "5"]},
        {"type": "line", "start": ["5", "5"], "end": ["0", "5"]},
        {"type": "line", "start": ["0", "5"], "end": ["0", "0"]},
        {"type": "line", "start": ["0", "0"], "end": ["-5", "0"]},
        {"type": "line", "start": ["-5", "0"], "end": ["-5", "-5"]},
        {"type": "line", "start": ["-5", "-5"], "end": ["0", "-5"]},
        {"type": "line", "start": ["0", "-5"], "end": ["0", "0"]},
    ], "tool_radius": "0.25", "side": "left"})
    assert code == 409
    assert body["error"]["code"] in {
        "non_adjacent_contact", "self_intersection",
        "trim_order_reversal", "adjacent_extra_contact",
    }


# ----------------------------------------------------------- exact encoding
def test_exact_expression_is_recomputable(client):
    code, body = post(client, {
        "contour": [
            {"type": "arc", "start": ["5", "0"], "end": ["-5", "0"],
             "center": ["0", "0"], "sense": "ccw"},
            {"type": "line", "start": ["-5", "0"], "end": ["5", "0"]},
        ],
        "tool_radius": "1", "side": "left", "decimal_places": 6,
    })
    assert code == 200
    line = next(p for p in body["compensated_contour"]["pieces"]
                if p["type"] == "line")
    x = line["start"]["x"]
    # sqrt(15) exact form: nested {"sqrt": {"rational": [15, 1]}}.
    assert "sqrt" in str(x["exact"])
    assert x["decimal"].lstrip("-").startswith("3.872983")
