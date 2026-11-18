"""No "parameters" in the metadata: the schema is inferred from the signature."""

from typing import Literal

from langgraph_json.metadata import AI_METADATA

_FACTORS = {"m": 1.0, "km": 1000.0, "ft": 0.3048, "nm": 1852.0}


@AI_METADATA({
    "name": "convert_distance",
    "domain": "utils",
    "read_only": True,
    "description": "Converts a distance between units (m, km, ft, nm).",
})
def convert_distance(value: float, from_unit: Literal["m", "km", "ft", "nm"], to_unit: Literal["m", "km", "ft", "nm"] = "m") -> dict:
    result = value * _FACTORS[from_unit] / _FACTORS[to_unit]
    return {"value": round(result, 6), "unit": to_unit}
