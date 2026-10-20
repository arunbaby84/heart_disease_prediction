from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as PDFImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SCREENSHOTS = DOCS / "screenshots"
OUTPUT = DOCS / "heart_disease_mlops_report.pdf"

INK = colors.HexColor("#18323A")
TEAL = colors.HexColor("#176B68")
PALE = colors.HexColor("#E8F2EF")
MUTED = colors.HexColor("#5B6D70")


def load_font(size: int, mono: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"]
        if mono
        else ["C:/Windows/Fonts/seguiemj.ttf", "C:/Windows/Fonts/arial.ttf"]
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def make_source_evidence(source: Path, output: Path, title: str, needles: tuple[str, ...]) -> None:
    source_lines = source.read_text(encoding="utf-8").splitlines()
    selected = [line for line in source_lines if any(needle in line for needle in needles)]
    excerpt = source_lines[:8] + ["..."] + selected
    wrapped = [part for line in excerpt for part in textwrap.wrap(line, width=92) or [""]]
    width = 1240
    line_height = 29
    image = Image.new("RGB", (width, 126 + line_height * len(wrapped)), "#F3F6F5")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 76), fill="#18323A")
    draw.text((32, 18), title, font=load_font(28), fill="#FFFFFF")
    draw.text((32, 88), f"Source: {source.relative_to(ROOT).as_posix()}", font=load_font(18), fill="#176B68")
    y = 120
    code_font = load_font(18, mono=True)
    for line in wrapped:
        draw.text((34, y), line, font=code_font, fill="#24363A")
        y += line_height
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def make_architecture(output: Path) -> None:
    width, height = 1500, 760
    image = Image.new("RGB", (width, height), "#F3F6F5")
    draw = ImageDraw.Draw(image)
    title_font = load_font(32)
    box_font = load_font(22)
    small_font = load_font(17)
    draw.text((48, 32), "HEART DISEASE MLOPS | SYSTEM ARCHITECTURE", font=title_font, fill="#18323A")

    boxes = [
        (50, 150, 345, 300, "UCI dataset", "Cleveland CSV"),
        (410, 150, 705, 300, "Data + EDA", "Download, clean, inspect"),
        (770, 150, 1065, 300, "Training", "Sklearn pipelines + MLflow"),
        (1130, 150, 1445, 300, "Model artifact", "Joblib / pickle"),
        (770, 470, 1065, 625, "FastAPI", "/predict  /health  /metrics"),
        (1130, 470, 1445, 625, "Runtime", "Docker Compose / Kubernetes"),
        (50, 470, 345, 625, "CI workflow", "Lint, test, train, archive"),
        (410, 470, 705, 625, "Observability", "Prometheus + Grafana + logs"),
    ]
    for x1, y1, x2, y2, heading, detail in boxes:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=14, fill="#FFFFFF", outline="#176B68", width=3)
        draw.text((x1 + 18, y1 + 40), heading, font=box_font, fill="#18323A")
        draw.text((x1 + 18, y1 + 88), detail, font=small_font, fill="#5B6D70")

    def arrow(start: tuple[int, int], end: tuple[int, int]) -> None:
        draw.line((*start, *end), fill="#176B68", width=5)
        draw.polygon([end, (end[0] - 13, end[1] - 8), (end[0] - 13, end[1] + 8)], fill="#176B68")

    def down_arrow(start: tuple[int, int], end: tuple[int, int]) -> None:
        draw.line((*start, *end), fill="#176B68", width=5)
        draw.polygon([end, (end[0] - 8, end[1] - 13), (end[0] + 8, end[1] - 13)], fill="#176B68")

    arrow((345, 225), (410, 225))
    arrow((705, 225), (770, 225))
    arrow((1065, 225), (1130, 225))
    down_arrow((1285, 300), (1285, 470))
    arrow((1130, 550), (1065, 550))
    arrow((770, 550), (705, 550))
    draw.text((50, 690), "CI validates and archives outputs; Kubernetes deployment is configured separately.", font=small_font, fill="#5B6D70")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25, leading=31, textColor=INK, alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=11, leading=16, textColor=MUTED, alignment=TA_CENTER, spaceAfter=16))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontSize=16, leading=20, textColor=TEAL, spaceBefore=8, spaceAfter=7))
    styles.add(ParagraphStyle(name="Subsection", parent=styles["Heading2"], fontSize=11, leading=14, textColor=INK, spaceBefore=7, spaceAfter=4))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=9, leading=13, textColor=INK, spaceAfter=6))
    styles.add(ParagraphStyle(name="CodeBlock", parent=styles["Code"], fontName="Courier", fontSize=8, leading=11, backColor=colors.HexColor("#F1F5F4"), borderPadding=7, textColor=INK, spaceBefore=3, spaceAfter=8))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=8, leading=10, textColor=MUTED, alignment=TA_CENTER, spaceAfter=5))
    return styles


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def fit_image(path: Path, width: float, height: float) -> PDFImage:
    image = PDFImage(str(path))
    ratio = min(width / image.imageWidth, height / image.imageHeight)
    image.drawWidth = image.imageWidth * ratio
    image.drawHeight = image.imageHeight * ratio
    return image


def add_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8E3E0"))
    canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "Heart Disease Prediction MLOps | Project report")
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_pdf() -> None:
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    make_architecture(DOCS / "architecture.png")
    make_source_evidence(
        ROOT / ".github" / "workflows" / "ci.yml",
        SCREENSHOTS / "ci_workflow_evidence.png",
        "GitHub Actions CI workflow | source evidence",
        ("push:", "pull_request:", "name: Set up", "name: Install", "name: Lint", "name: Download", "name: Run unit", "name: Train", "name: Upload"),
    )
    make_source_evidence(
        ROOT / "k8s" / "deployment.yaml",
        SCREENSHOTS / "deployment_workflow_evidence.png",
        "Kubernetes deployment manifest | source evidence",
        ("kind: Deployment", "replicas:", "image:", "readinessProbe:", "livenessProbe:", "kind: Service", "type: LoadBalancer"),
    )

    styles = make_styles()
    body = styles["BodySmall"]
    story = [
        Spacer(1, 24 * mm),
        para("Heart Disease Prediction", styles["CoverTitle"]),
        para("MLOps project report | Cleveland dataset", styles["Subtitle"]),
        para('<link href="https://github.com/arunbaby84/heart_disease_prediction" color="#176B68">github.com/arunbaby84/heart_disease_prediction</link>', styles["Subtitle"]),
        Spacer(1, 7 * mm),
        para("1. Setup and installation", styles["Section"]),
        para("Requirements: Python 3.11 or newer. Docker Desktop is optional for local services; a Kubernetes cluster is required to apply the deployment manifest.", body),
        para("Windows PowerShell", styles["Subsection"]),
        para("py -3.12 -m venv .venv<br/>.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip<br/>.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt", styles["CodeBlock"]),
        para("Linux / macOS", styles["Subsection"]),
        para("python3 -m venv .venv<br/>source .venv/bin/activate<br/>python -m pip install --upgrade pip<br/>python -m pip install -r requirements.txt", styles["CodeBlock"]),
        para("Download, preprocess, train, and serve", styles["Subsection"]),
        para("For Windows, run each command through .venv\\Scripts\\python.exe instead of python.", body),
        para("python scripts/download_data.py<br/>python scripts/preprocess_data.py<br/>python scripts/train_model.py<br/>python -m uvicorn app.main:app --host 0.0.0.0 --port 8000", styles["CodeBlock"]),
        para("Run docker compose up --build for the API, MLflow, Prometheus, and Grafana services. Local service ports are 8000, 5000, 9090, and 3000 respectively.", body),
        PageBreak(),
        para("2. Dataset, EDA, and modeling", styles["Section"]),
        para("The downloader reads the Cleveland heart disease CSV from the UCI archive, converts '?' to missing values, coerces columns to numeric, drops incomplete rows, and maps target values above zero to the positive class. EDA covers distributions, class balance, correlations, pairwise feature relationships, and IQR outliers.", body),
        fit_image(SCREENSHOTS / "eda_summary.png", 170 * mm, 75 * mm),
        para("EDA overview: age, cholesterol, maximum heart rate, and class balance.", styles["Caption"]),
        para("Candidate pipelines combine median imputation and standard scaling with Logistic Regression or a class-weighted Random Forest (300 trees, minimum leaf size 2). A fixed-seed stratified 80/20 split supplies holdout metrics; five-fold stratified cross-validation on training data estimates ROC-AUC stability.", body),
    ]

    metrics = json.loads((ROOT / "artifacts" / "model_report.json").read_text(encoding="utf-8"))
    table_data = [["Model", "Accuracy", "Precision", "Recall", "Holdout AUC", "CV AUC mean (std)"]]
    for name, label in (("logistic_regression", "Logistic Regression"), ("random_forest", "Random Forest")):
        values = metrics[name]
        table_data.append([
            label,
            f"{values['accuracy']:.3f}",
            f"{values['precision']:.3f}",
            f"{values['recall']:.3f}",
            f"{values['roc_auc']:.3f}",
            f"{values['cv_roc_auc_mean']:.3f} ({values['cv_roc_auc_std']:.3f})",
        ])
    table = Table(table_data, colWidths=[40 * mm, 22 * mm, 22 * mm, 19 * mm, 23 * mm, 40 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8E3E0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([
        Spacer(1, 4 * mm),
        table,
        Spacer(1, 3 * mm),
        para("Current selection uses holdout ROC-AUC, so Logistic Regression is selected by the recorded metrics. Random Forest leads on accuracy, precision, recall, and mean cross-validation ROC-AUC. The small dataset and narrow holdout difference make these educational results, not clinical evidence.", body),
        fit_image(SCREENSHOTS / "feature_correlation.png", 142 * mm, 91 * mm),
        para("Feature correlation matrix.", styles["Caption"]),
        PageBreak(),
        para("3. Experiment tracking and architecture", styles["Section"]),
        para("Each training execution logs two candidate runs to the heart_disease_prediction MLflow experiment. Runs include estimator parameters, holdout metrics, cross-validation ROC-AUC mean and standard deviation, and a serialized scikit-learn pipeline with inferred signature. The local store is mlruns/. Model files are artifacts/best_model.joblib and artifacts/best_model.pkl; metrics are summarized in artifacts/model_report.json. Browse runs with python -m mlflow ui --host 127.0.0.1 --port 5000.", body),
        fit_image(DOCS / "architecture.png", 175 * mm, 104 * mm),
        para("Architecture and workflow boundaries. CI validates and archives outputs; cluster deployment is configured separately.", styles["Caption"]),
        para("The FastAPI service shares the saved preprocessing-and-estimator pipeline. /predict returns a class and positive-class probability; /health supports probes; /metrics exposes Prometheus instrumentation. Middleware logs request ID, method, path, status, and duration.", body),
        PageBreak(),
        para("4. CI/CD and deployment", styles["Section"]),
        para("GitHub Actions runs for pushes and pull requests to main or master. It sets up Python 3.11, installs requirements, runs Ruff, downloads data, runs pytest, and trains the models. Workflow reports, model outputs, MLflow runs, and processed data are uploaded as artifacts. The workflow does not publish a container image or automatically deploy to Kubernetes.", body),
        fit_image(SCREENSHOTS / "ci_workflow_evidence.png", 174 * mm, 81 * mm),
        para("Configuration-derived source evidence, not a screenshot of a live GitHub Actions run.", styles["Caption"]),
        para("Kubernetes configuration requests two API replicas, uses /health for readiness and liveness probes, and exposes port 8000 through a LoadBalancer Service. After making the referenced image available to a configured cluster, apply it with kubectl apply -f k8s/deployment.yaml.", body),
        fit_image(SCREENSHOTS / "deployment_workflow_evidence.png", 174 * mm, 81 * mm),
        para("Configuration-derived source evidence, not a screenshot of a running cluster.", styles["Caption"]),
        PageBreak(),
        para("5. Reproducibility and limitations", styles["Section"]),
        para("Pinned dependencies and fixed random seeds make the workflow repeatable. The saved full pipeline keeps imputation and scaling consistent between training and inference. Run tests with python -m pytest -q. Docker Compose includes MLflow, the API, Prometheus, and Grafana with provisioned dashboards.", body),
        para("Evidence status", styles["Subsection"]),
        para("The repository includes EDA figures but no captured live GitHub Actions run or deployed-cluster dashboard. The two workflow images in this report are labeled renderings of checked-in configuration, not fabricated operational screenshots. Capture real run and cluster screens from the target GitHub repository and environment if the assignment requires proof of a successful live run or deployment.", body),
        para("This dataset and model are for educational demonstration only. Any real healthcare use would require external validation, calibration, subgroup analysis, privacy and security review, and clinical governance.", body),
        Spacer(1, 5 * mm),
        para('<b>Repository:</b> <link href="https://github.com/arunbaby84/heart_disease_prediction" color="#176B68">https://github.com/arunbaby84/heart_disease_prediction</link>', body),
    ])

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=21 * mm,
        title="Heart Disease Prediction MLOps Report",
        author="Heart Disease Prediction MLOps Project",
    )
    document.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build_pdf()