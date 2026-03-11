"""
pdf_export.py
-------------
Generates a downloadable PDF Water Quality Report.
"""

from fpdf import FPDF
import os
import tempfile
from datetime import datetime


class WaterQualityPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(30, 100, 180)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, "  Water Quality Analysis Report", ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Page {self.page_no()}", align="C")


def generate_pdf(prediction: int, probability: float,
                 inputs: dict, param_analysis: list,
                 risk: dict, ai_report: str) -> bytes:
    """
    Creates a PDF report and returns its bytes for download.
    """
    pdf = WaterQualityPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Prediction Banner ──────────────────────────────────────────────────────
    status = "SAFE FOR DRINKING" if prediction == 1 else "NOT SAFE FOR DRINKING"
    color  = (34, 139, 34) if prediction == 1 else (200, 30, 30)

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*color)
    pdf.cell(0, 12, f"Prediction: {status}", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Model Confidence: {round(probability * 100, 1)}%", ln=True, align="C")
    pdf.cell(0, 8, f"Risk Score: {risk['score']} / 100  |  {risk['level']}", ln=True, align="C")
    pdf.ln(5)

    # ── Parameter Table ───────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(220, 235, 255)
    pdf.cell(0, 9, "Parameter Analysis", ln=True, fill=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(75, 8, "Parameter", border=1, fill=True)
    pdf.cell(35, 8, "Value", border=1, fill=True)
    pdf.cell(50, 8, "Safe Range", border=1, fill=True)
    pdf.cell(30, 8, "Status", border=1, fill=True, ln=True)

    pdf.set_font("Helvetica", "", 10)
    for p in param_analysis:
        safe_range = f"{p['safe_low']} – {p['safe_high']}"
        status_str = p["status"]

        if status_str == "Unsafe":
            pdf.set_text_color(200, 30, 30)
        else:
            pdf.set_text_color(34, 139, 34)

        pdf.cell(75, 7, p["label"], border=1)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(35, 7, str(round(p["value"], 3)), border=1)
        pdf.cell(50, 7, safe_range, border=1)

        if status_str == "Unsafe":
            pdf.set_text_color(200, 30, 30)
        else:
            pdf.set_text_color(34, 139, 34)

        pdf.cell(30, 7, status_str, border=1, ln=True)
        pdf.set_text_color(0, 0, 0)

    pdf.ln(6)

    # ── AI Report ─────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(220, 235, 255)
    pdf.cell(0, 9, "AI-Generated Water Quality Analysis", ln=True, fill=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    # Clean and write the AI report text
    for line in ai_report.split("\n"):
        line = line.replace("–", "-").replace("—", "-").replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue
        # Check if it's a section heading
        import re
        if re.match(r"^\d+\.", line):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 7, line)
            pdf.set_font("Helvetica", "", 10)
        else:
            pdf.multi_cell(0, 6, line)

    # ── Save to temp and return bytes ──────────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name

    pdf.output(tmp_path)

    with open(tmp_path, "rb") as f:
        pdf_bytes = f.read()

    os.remove(tmp_path)
    return pdf_bytes
