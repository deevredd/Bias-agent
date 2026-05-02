import io
import base64
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.io as pio
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


SEVERITY_COLORS = {
    "high":    colors.HexColor("#E53E3E"),
    "medium":  colors.HexColor("#DD6B20"),
    "low":     colors.HexColor("#38A169"),
    "unknown": colors.HexColor("#718096"),
}

SEVERITY_EMOJI = {
    "high": "HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
    "unknown": "UNKNOWN",
}


def _chart_to_image(fig, width=400, height=250):
    """Convert a plotly figure to a ReportLab Image flowable."""
    img_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=2)
    buf = io.BytesIO(img_bytes)
    return RLImage(buf, width=14*cm, height=8*cm)


def _bias_type_label(bias_type: str) -> str:
    return bias_type.replace("_", " ").title()


def generate_pdf_report(
    profile: dict,
    bias_results: list,
    dataset_name: str = "Dataset",
    impact_results: dict = None,
) -> bytes:
    """
    Generate a full PDF bias report.
    Returns raw PDF bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    style_normal = styles["Normal"]
    style_normal.fontSize = 9
    style_normal.leading = 14

    style_h1 = ParagraphStyle("H1", fontSize=20, fontName="Helvetica-Bold",
                               spaceAfter=6, textColor=colors.HexColor("#1A202C"))
    style_h2 = ParagraphStyle("H2", fontSize=13, fontName="Helvetica-Bold",
                               spaceAfter=4, spaceBefore=12,
                               textColor=colors.HexColor("#2D3748"))
    style_h3 = ParagraphStyle("H3", fontSize=10, fontName="Helvetica-Bold",
                               spaceAfter=3, spaceBefore=8,
                               textColor=colors.HexColor("#4A5568"))
    style_small = ParagraphStyle("Small", fontSize=8, textColor=colors.HexColor("#718096"),
                                  leading=12)
    style_rec = ParagraphStyle("Rec", fontSize=9, leading=13,
                                textColor=colors.HexColor("#2B6CB0"),
                                backColor=colors.HexColor("#EBF8FF"),
                                borderPadding=6, leftIndent=6)

    story = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Automated Bias Detection Report", style_h1))
    story.append(Paragraph(
        f"Dataset: <b>{dataset_name}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}",
        style_small
    ))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#E2E8F0"), spaceAfter=12))

    # ── Dataset Summary ───────────────────────────────────────────────────────
    story.append(Paragraph("Dataset Overview", style_h2))
    shape = profile.get("shape", {})
    protected = profile.get("suspected_protected_attrs", [])
    datetime_cols = profile.get("suspected_datetime_cols", [])

    summary_data = [
        ["Rows", str(shape.get("rows", "—"))],
        ["Columns", str(shape.get("cols", "—"))],
        ["Protected attributes detected", ", ".join(protected) if protected else "None"],
        ["Datetime columns detected", ", ".join(datetime_cols) if datetime_cols else "None"],
    ]
    summary_table = Table(summary_data, colWidths=[6*cm, 10*cm])
    summary_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.HexColor("#F7FAFC"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Bias Score Card ───────────────────────────────────────────────────────
    story.append(Paragraph("Bias Score Summary", style_h2))
    score_header = [["Bias Type", "Severity", "Detected", "Affected Columns"]]
    score_rows = []
    for r in bias_results:
        score_rows.append([
            _bias_type_label(r["bias_type"]),
            r["severity"].upper(),
            "Yes" if r["detected"] else "No",
            ", ".join(r["affected_columns"][:4]) if r["affected_columns"] else "—",
        ])
    score_table = Table(score_header + score_rows,
                        colWidths=[5*cm, 3*cm, 2.5*cm, 6.5*cm])
    severity_colors_map = [
        SEVERITY_COLORS.get(r["severity"], colors.grey) for r in bias_results
    ]
    ts = TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2D3748")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])
    for i, sev_color in enumerate(severity_colors_map):
        ts.add("TEXTCOLOR", (1, i+1), (1, i+1), sev_color)
        ts.add("FONTNAME", (1, i+1), (1, i+1), "Helvetica-Bold")
    score_table.setStyle(ts)
    story.append(score_table)
    story.append(PageBreak())

    # ── Detailed Findings ─────────────────────────────────────────────────────
    story.append(Paragraph("Detailed Findings", style_h1))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#E2E8F0"), spaceAfter=8))

    for result in bias_results:
        bias_label = _bias_type_label(result["bias_type"])
        severity = result["severity"]
        sev_color = SEVERITY_COLORS.get(severity, colors.grey)

        story.append(Paragraph(
            f'{bias_label} — <font color="#{sev_color.hexval()[2:]}">'
            f'{severity.upper()}</font>',
            style_h2
        ))

        if result["affected_columns"]:
            story.append(Paragraph(
                f"<b>Affected columns:</b> {', '.join(result['affected_columns'])}",
                style_normal
            ))
            story.append(Spacer(1, 0.2*cm))

        # Recommendation box
        story.append(Paragraph(result["recommendation"], style_rec))
        story.append(Spacer(1, 0.3*cm))

        # Explainability
        explain = result.get("explainability", {})
        if explain.get("summary"):
            story.append(Paragraph("<b>Why this matters:</b>", style_h3))
            story.append(Paragraph(explain["summary"], style_normal))
            story.append(Spacer(1, 0.2*cm))

        if explain.get("examples"):
            for ex in explain["examples"]:
                story.append(Paragraph(f"• {ex}", style_normal))
            story.append(Spacer(1, 0.2*cm))

        # Evidence details
        evidence = result.get("evidence", {})

        # Demographic — chart
        if result["bias_type"] == "demographic_disparity":
            for col, findings in evidence.items():
                if not isinstance(findings, dict) or "observed" not in findings:
                    continue
                observed = findings["observed"]
                ref = findings.get("reference_distribution", {})
                rows = []
                all_groups = sorted(set(list(observed.keys()) + list(ref.keys())))
                for g in all_groups:
                    if g in observed:
                        rows.append({"Group": g, "Rate": observed[g], "Source": "Observed"})
                    if g in ref:
                        rows.append({"Group": g, "Rate": ref[g], "Source": "Reference"})
                if rows:
                    try:
                        fig = px.bar(
                            pd.DataFrame(rows), x="Group", y="Rate",
                            color="Source", barmode="group",
                            title=f"{col} — Observed vs Reference",
                            color_discrete_map={"Observed": "#4C9BE8", "Reference": "#E8854C"},
                        )
                        fig.update_layout(
                            plot_bgcolor="white", paper_bgcolor="white",
                            font=dict(size=10), margin=dict(l=20, r=20, t=40, b=20)
                        )
                        story.append(_chart_to_image(fig))
                    except Exception:
                        pass

        # Temporal — table
        elif result["bias_type"] == "temporal_bias":
            drift_items = {k: v for k, v in evidence.items() if "_drift" in k}
            if drift_items:
                story.append(Paragraph("<b>Feature drift detected:</b>", style_h3))
                t_header = [["Column", "KS Stat", "P Value", "Early Mean", "Late Mean"]]
                t_rows = []
                for key, val in drift_items.items():
                    t_rows.append([
                        key.replace("_drift", ""),
                        str(val.get("ks_stat", "")),
                        str(val.get("p_value", "")),
                        str(val.get("early_mean", "")),
                        str(val.get("late_mean", "")),
                    ])
                t = Table(t_header + t_rows,
                          colWidths=[4*cm, 2.5*cm, 2.5*cm, 3*cm, 3*cm])
                t.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2D3748")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.HexColor("#F7FAFC"), colors.white]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(t)

        # Selection — text bullets
        else:
            for key, val in evidence.items():
                if isinstance(val, dict) and "interpretation" in val:
                    story.append(Paragraph(f"• {val['interpretation']}", style_normal))

        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                 color=colors.HexColor("#E2E8F0"), spaceAfter=4))

    # ── Impact Simulation ─────────────────────────────────────────────────────
    if impact_results:
        story.append(PageBreak())
        story.append(Paragraph("Model Impact Simulation", style_h1))
        story.append(HRFlowable(width="100%", thickness=1,
                                 color=colors.HexColor("#E2E8F0"), spaceAfter=8))
        story.append(Paragraph(
            "Comparison of model performance trained on the original biased dataset "
            "versus the balanced dataset.",
            style_normal
        ))
        story.append(Spacer(1, 0.3*cm))

        imp_header = [["Metric", "Biased Dataset", "Balanced Dataset", "Δ Change"]]
        imp_rows = []
        biased_overall = impact_results.get("biased", {}).get("overall", {})
        balanced_overall = impact_results.get("balanced", {}).get("overall", {})
        for metric in ["accuracy", "precision", "recall", "f1"]:
            b_val = biased_overall.get(metric, 0)
            bal_val = balanced_overall.get(metric, 0)
            delta = round(bal_val - b_val, 4) if b_val and bal_val else "—"
            delta_str = f"+{delta}" if isinstance(delta, float) and delta > 0 else str(delta)
            imp_rows.append([metric.title(), str(b_val), str(bal_val), delta_str])

        imp_table = Table(imp_header + imp_rows,
                          colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
        imp_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2D3748")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#F7FAFC"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(imp_table)

    # ── Footer note ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#E2E8F0"), spaceAfter=4))
    story.append(Paragraph(
        "Generated by Automated Bias Detector — for portfolio and research use.",
        style_small
    ))

    doc.build(story)
    return buf.getvalue()