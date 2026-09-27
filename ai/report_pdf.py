"""
VOLTURA Investigation PDF Report Builder
-----------------------------------------
Converts the selected anomaly event data + Gemini AI analysis into a
professionally formatted PDF using ReportLab.

The PDF filename follows the convention:
    VOLTURA_<EVENT_ID>_Investigation_Report.pdf
"""

import re
from html import escape
from io import BytesIO
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY    = colors.HexColor("#19324a")
TEAL    = colors.HexColor("#1a6e8a")
LIGHT   = colors.HexColor("#f2f5f7")
STRIPE  = colors.HexColor("#e8edf2")
BORDER  = colors.HexColor("#b8c4cf")
WHITE   = colors.white
BLACK   = colors.HexColor("#1a1a2e")

SEVERITY_COLORS = {
    "NORMAL":      colors.HexColor("#00cc66"),
    "WATCH":       colors.HexColor("#e6b800"),
    "INVESTIGATE": colors.HexColor("#ff8c00"),
    "HIGH":        colors.HexColor("#e63946"),
    "CRITICAL":    colors.HexColor("#9b0000"),
}


def _value(value: Any) -> str:
    """Return 'Not available' for None, else str(value)."""
    return "Not available" if value is None else str(value)


def _na_if_placeholder(v: Any) -> str:
    s = _value(v)
    return "Not available" if s in ("None", "", "nan") else s


# ── Markdown-to-ReportLab helpers ──────────────────────────────────────────────

def _md_to_rl(text: str) -> str:
    """
    Convert a subset of markdown inline formatting to ReportLab XML.
    Handles **bold**, *italic*, and escapes HTML special characters.
    """
    # Escape HTML first so that < > & are safe in Paragraph XML
    text = escape(text)
    # **bold**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # *italic* (single asterisk that wasn't already consumed)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def _add_analysis_text(story, analysis: str, styles):
    """
    Parse the Gemini markdown analysis and add to the ReportLab story.
    Handles:
      ### headings   → Heading3 style
      **bold**       → bold inline text
      - bullet       → indented bullet paragraph
      numbered. list → indented paragraph
      blank lines    → small spacer
      normal text    → BodyText paragraph
    """
    h3_style     = styles["Heading3"]
    body_style   = styles["BodyText"]
    bullet_style = ParagraphStyle(
        "BulletBody",
        parent=body_style,
        leftIndent=18,
        bulletIndent=6,
        spaceAfter=2,
    )

    for raw_line in analysis.splitlines():
        line = raw_line.strip()

        if not line:
            story.append(Spacer(1, 4))
            continue

        if line.startswith("### "):
            heading_text = _md_to_rl(line[4:])
            story.append(Spacer(1, 6))
            story.append(Paragraph(heading_text, h3_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
            story.append(Spacer(1, 3))

        elif line.startswith("## "):
            heading_text = _md_to_rl(line[3:])
            story.append(Spacer(1, 8))
            story.append(Paragraph(heading_text, styles["Heading2"]))
            story.append(Spacer(1, 3))

        elif line.startswith("- ") or line.startswith("* "):
            content = _md_to_rl(line[2:])
            story.append(Paragraph(f"&bull; {content}", bullet_style))

        elif re.match(r"^\d+\.\s", line):
            content = _md_to_rl(re.sub(r"^\d+\.\s", "", line))
            story.append(Paragraph(f"&bull; {content}", bullet_style))

        else:
            content = _md_to_rl(line)
            story.append(Paragraph(content, body_style))
            story.append(Spacer(1, 2))


def _evidence_table(rows, col_widths, styles) -> Table:
    """Build a styled two-column evidence table."""
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  9),
        ("GRID",          (0, 0), (-1, -1), 0.35, BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, STRIPE]),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("LEADING",       (0, 0), (-1, -1), 13),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    return table


# ── Public API ─────────────────────────────────────────────────────────────────

def build_investigation_pdf(
    event: Dict[str, Any],
    analysis: str,
    tariff: float,
    intervention_cost: float,
) -> bytes:
    """
    Build a full investigation PDF for the selected anomaly event.

    Sections:
      1. Event Information
      2. Energy Analysis
      3. Operating Conditions
      4. AI/ML Evidence
      5. Gemini AI Investigation (all 11 sections)

    Returns raw PDF bytes.  Filename to be set by the caller as:
        VOLTURA_<EVENT_ID>_Investigation_Report.pdf
    """
    context      = event.get("row_data") or {}
    actual       = event.get("actual_energy")
    expected     = event.get("expected_energy")
    deviation    = (actual - expected) if (actual is not None and expected is not None) else None
    excess       = max(0.0, deviation) if deviation is not None else None
    savings      = (excess * tariff) if excess is not None else None
    roi          = (savings / intervention_cost * 100) if (savings is not None and intervention_cost) else None
    severity     = event.get("severity", "")
    event_id     = _na_if_placeholder(event.get("event_id"))

    # ── Document setup ────────────────────────────────────────────────────────
    buffer   = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=f"VOLTURA Investigation Report – {event_id}",
        author="VOLTURA AI Auditor",
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "NEXYRATitle",
        parent=styles["Title"],
        textColor=NAVY,
        fontSize=18,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "NEXYRASubtitle",
        parent=styles["Normal"],
        textColor=TEAL,
        fontSize=10,
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=NAVY,
        fontSize=11,
        spaceBefore=12,
        spaceAfter=4,
    )

    # Severity colour for header badge
    sev_color = SEVERITY_COLORS.get(severity, TEAL)

    story = []

    # ── Title block ───────────────────────────────────────────────────────────
    story.append(Paragraph("VOLTURA Investigation Report", title_style))
    story.append(Paragraph(
        f"Event&nbsp;&nbsp;<b>{event_id}</b>&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Equipment&nbsp;<b>{_na_if_placeholder(event.get('equipment_id'))}</b>&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Severity&nbsp;<font color='{sev_color.hexval()}'><b>{severity or 'N/A'}</b></font>",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL))
    story.append(Spacer(1, 10))

    col_w = [2.3 * inch, 4.7 * inch]

    # ── Section 1: Event Information ──────────────────────────────────────────
    story.append(Paragraph("Event Information", section_style))
    event_rows = [
        ["Parameter", "Value"],
        ["Event ID",       _na_if_placeholder(event.get("event_id"))],
        ["Equipment ID",   _na_if_placeholder(event.get("equipment_id"))],
        ["Timestamp",      _na_if_placeholder(event.get("timestamp"))],
        ["Severity",       _na_if_placeholder(event.get("severity"))],
        ["Duration",       f"{_na_if_placeholder(event.get('duration'))} observation(s)"],
    ]
    story.append(_evidence_table(event_rows, col_w, styles))
    story.append(Spacer(1, 10))

    # ── Section 2: Energy Analysis ────────────────────────────────────────────
    story.append(Paragraph("Energy Analysis", section_style))
    energy_rows = [
        ["Parameter", "Value"],
        ["Actual Energy",    f"{_na_if_placeholder(actual)} kWh"],
        ["Expected Energy",  f"{_na_if_placeholder(expected)} kWh"],
        ["Energy Deviation", f"{_na_if_placeholder(deviation)} kWh"],
        ["Deviation %",      f"{_na_if_placeholder(event.get('deviation_pct'))}%"],
        ["Excess Energy",    f"{_na_if_placeholder(excess)} kWh"],
    ]
    story.append(_evidence_table(energy_rows, col_w, styles))
    story.append(Spacer(1, 10))

    # ── Section 3: Operating Conditions ──────────────────────────────────────
    story.append(Paragraph("Operating Conditions", section_style))
    conditions_rows = [
        ["Parameter", "Value"],
        ["Building Load",             f"{_na_if_placeholder(context.get('Building Load (RT)'))} RT"],
        ["Chilled Water Rate",        f"{_na_if_placeholder(context.get('Chilled Water Rate (L/sec)'))} L/sec"],
        ["Cooling Water Temperature", f"{_na_if_placeholder(context.get('Cooling Water Temperature (C)'))} °C"],
        ["Outside Temperature",       f"{_na_if_placeholder(context.get('Outside Temperature (F)'))} °F"],
        ["Dew Point",                 f"{_na_if_placeholder(context.get('Dew Point (F)'))} °F"],
        ["Humidity",                  f"{_na_if_placeholder(context.get('Humidity (%)'))}%"],
        ["Wind Speed",                f"{_na_if_placeholder(context.get('Wind Speed (mph)'))} mph"],
        ["Pressure",                  f"{_na_if_placeholder(context.get('Pressure (in)'))} in"],
    ]
    story.append(_evidence_table(conditions_rows, col_w, styles))
    story.append(Spacer(1, 10))

    # ── Section 4: AI/ML Evidence ─────────────────────────────────────────────
    story.append(Paragraph("AI / ML Evidence", section_style))
    ml_rows = [
        ["Parameter", "Value"],
        ["Context / Isolation Forest Score", _na_if_placeholder(event.get("context_score"))],
        ["Anomaly Decision",                 _na_if_placeholder(event.get("is_anomaly"))],
        ["Severity",                         _na_if_placeholder(event.get("severity"))],
        ["Reason",                           _na_if_placeholder(event.get("reason"))],
    ]
    story.append(_evidence_table(ml_rows, col_w, styles))
    story.append(Spacer(1, 10))

    # ── Section 5: Financial Data ─────────────────────────────────────────────
    story.append(Paragraph("Financial Data", section_style))
    fin_rows = [
        ["Parameter", "Value"],
        ["Electricity Tariff",  f"Rs {tariff:.2f} / kWh"],
        ["Estimated Savings",   f"Rs {_na_if_placeholder(round(savings, 2) if savings is not None else None)}"],
        ["Intervention Cost",   f"Rs {intervention_cost:,.2f}"],
        ["Estimated ROI",       f"{_na_if_placeholder(round(roi, 1) if roi is not None else None)}%  (not a guaranteed return)"],
    ]
    story.append(_evidence_table(fin_rows, col_w, styles))
    story.append(Spacer(1, 14))

    # ── Section 6: Gemini AI Investigation ────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Gemini AI Investigation Analysis", section_style))
    story.append(Spacer(1, 4))

    if analysis:
        _add_analysis_text(story, analysis, styles)
    else:
        story.append(Paragraph("No AI analysis available for this event.", styles["BodyText"]))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<i>This report was automatically generated by VOLTURA AI Auditor. "
        "AI analysis is advisory only and requires verification by a qualified engineer.</i>",
        ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=7.5, textColor=colors.HexColor("#888888")),
    ))

    document.build(story)
    return buffer.getvalue()
