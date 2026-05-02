import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from typing import Optional

from core.loader import load_dataset
from core.profiler import DataProfiler
from core.bias_detectors import run_all_detectors
from core.synthetic import balance_with_smote, balance_demographic_group
from core.impact import simulate_impact
from core.explainability import explain_bias_results
from core.report import generate_pdf_report
from core.history import save_scan_result, load_history, list_tracked_datasets, build_trend_dataframe
from core.reference_store import save_custom_reference, load_all_references


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):
            return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def clean(data):
    return json.loads(json.dumps(data, cls=NumpyEncoder))


app = FastAPI(title="Bias Detector API", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_store = {}


# ── Core scan ──────────────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    snapshot_label: Optional[str] = Query(default=None),
):
    try:
        contents = await file.read()
        df = load_dataset(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not load file: {e}")

    try:
        profile = DataProfiler(df).run()
    except Exception as e:
        raise HTTPException(500, f"Profiling failed: {e}")

    try:
        all_refs = load_all_references()
        bias_results = run_all_detectors(df, profile, references=all_refs)
        bias_results = explain_bias_results(df, bias_results, profile)
    except Exception as e:
        raise HTTPException(500, f"Bias detection failed: {e}")

    session_id = file.filename
    _store[session_id] = df
    _store[f"{session_id}_scan"] = {
        "profile": profile,
        "bias_results": bias_results,
    }

    # Save to history
    try:
        save_scan_result(
            dataset_name=file.filename,
            bias_results=bias_results,
            profile=profile,
            label=snapshot_label or file.filename,
        )
    except Exception:
        pass

    return JSONResponse(content=clean({
        "session_id": session_id,
        "profile": profile,
        "bias_results": bias_results,
    }))


# ── PDF report ─────────────────────────────────────────────────────────────────
@app.get("/report/pdf")
async def pdf_report(
    session_id: str = Query(...),
    include_impact: bool = Query(default=False),
):
    if session_id not in _store:
        raise HTTPException(404, "Session not found. Re-upload the file.")

    session = _store.get(f"{session_id}_scan")
    if not session:
        raise HTTPException(404, "No scan results found. Run /analyze first.")

    impact = _store.get(f"{session_id}_impact") if include_impact else None

    try:
        pdf_bytes = generate_pdf_report(
            profile=session["profile"],
            bias_results=session["bias_results"],
            dataset_name=session_id,
            impact_results=impact,
        )
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {e}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="bias_report_{session_id}.pdf"'},
    )


# ── History / trend ────────────────────────────────────────────────────────────
@app.get("/history/datasets")
async def history_datasets():
    return {"datasets": list_tracked_datasets()}


@app.get("/history/trend")
async def history_trend(dataset_name: Optional[str] = Query(default=None)):
    records = load_history(dataset_name)
    if not records:
        return {"records": [], "trend": []}
    df_trend = build_trend_dataframe(records)
    return clean({
        "records": records,
        "trend": df_trend.to_dict(orient="records"),
    })


# ── Custom reference distributions ────────────────────────────────────────────
@app.get("/references")
async def get_references():
    return clean({"references": load_all_references()})


@app.post("/references/custom")
async def add_reference(
    name: str = Query(...),
    col_hint: str = Query(...),
    distribution: dict = Body(...),
):
    try:
        filepath = save_custom_reference(name, col_hint, distribution)
    except Exception as e:
        raise HTTPException(400, f"Could not save reference: {e}")
    return {"saved": True, "filepath": filepath, "name": name}


# ── Balance / impact (unchanged) ───────────────────────────────────────────────
@app.post("/balance/smote")
async def balance_smote(session_id: str = Query(...), target_col: str = Query(...)):
    df = _store.get(session_id)
    if df is None:
        raise HTTPException(404, "Session not found.")
    try:
        balanced = balance_with_smote(df, target_col)
        _store[f"{session_id}_balanced"] = balanced
    except Exception as e:
        raise HTTPException(400, str(e))
    return JSONResponse(content=clean({
        "original_shape": list(df.shape),
        "balanced_shape": list(balanced.shape),
        "synthetic_rows_added": int(balanced["is_synthetic"].sum()),
        "class_distribution": {str(k): int(v) for k, v in balanced[target_col].value_counts().items()},
    }))


@app.post("/balance/demographic")
async def balance_demographic(
    session_id: str = Query(...),
    group_col: str = Query(...),
    target_group: str = Query(...),
    target_count: int = Query(...),
):
    df = _store.get(session_id)
    if df is None:
        raise HTTPException(404, "Session not found.")
    try:
        balanced = balance_demographic_group(df, group_col, target_group, target_count)
        _store[f"{session_id}_balanced"] = balanced
    except Exception as e:
        raise HTTPException(400, str(e))
    return JSONResponse(content=clean({
        "original_count": int((df[group_col] == target_group).sum()),
        "new_count": int((balanced[group_col] == target_group).sum()),
        "synthetic_rows_added": int(balanced["is_synthetic"].sum()),
    }))


@app.post("/simulate-impact")
async def impact(
    session_id: str = Query(...),
    target_col: str = Query(...),
    protected_col: Optional[str] = Query(default=None),
):
    original = _store.get(session_id)
    balanced = _store.get(f"{session_id}_balanced")
    if original is None:
        raise HTTPException(404, "Session not found.")
    if balanced is None:
        raise HTTPException(404, "No balanced dataset found. Run /balance first.")
    try:
        results = simulate_impact(original, balanced, target_col, protected_col)
        _store[f"{session_id}_impact"] = results
    except Exception as e:
        raise HTTPException(500, f"Simulation failed: {e}")
    return JSONResponse(content=clean(results))


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}