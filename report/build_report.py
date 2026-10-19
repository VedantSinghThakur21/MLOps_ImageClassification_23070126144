"""Build the project report as Word (.docx) and PDF from screenshots/ and the repo files.

Run from the repository root with a Python that has python-docx:  python report/build_report.py
Screenshots that are missing are skipped, so new ones can be dropped into screenshots/ and the report rebuilt.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "screenshots"
OUT = ROOT / "report"
NAME = "MLOps_Report_23070126144"
REPO = "https://github.com/VedantSinghThakur21/MLOps_ImageClassification_23070126144"
APP_URL = "http://15.252.12.60/"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

for source, target in (
    ("artifacts/metrics/confusion_matrix.png", "19_confusion_matrix.png"),
    ("artifacts/model/training_history.png", "20_training_history.png"),
):
    if (ROOT / source).exists():
        shutil.copy(ROOT / source, SHOTS / target)

metrics = json.loads((ROOT / "artifacts/metrics/metrics.json").read_text()) if (ROOT / "artifacts/metrics/metrics.json").exists() else {
    "accuracy": 0.9067, "precision": 0.9075, "recall": 0.9067, "f1_score": 0.9066}

figure_count = 0


def code(path: str) -> tuple:
    return ("code", (ROOT / path).read_text(encoding="utf-8").rstrip())


H1, H2, P, BUL, NUM, CODE, IMG, TABLE = "h1", "h2", "p", "bullets", "numbers", "code", "img", "table"

CONTENT: list[tuple] = [
    (H1, "1. Introduction and Objectives"),
    (P, "This project builds, versions and deploys an image classification application that recognises six types of natural "
        "and urban scenes (buildings, forest, glacier, mountain, sea, street) from the Intel Image Dataset. A MobileNetV2 "
        "transfer learning model is trained through a reproducible DVC pipeline, its artifacts are versioned in AWS S3, "
        "and a Flask web application serving the model is containerised with Docker and deployed automatically to an AWS EC2 "
        "instance by a GitHub Actions CI/CD workflow."),
    (P, "Objectives covered in this report:"),
    (BUL, [
        "dvc.yaml defining the five-stage DVC pipeline (data collection, processing, feature extraction, training, evaluation).",
        "ci-cd.yaml implementing the GitHub Actions CI/CD workflow (.github/workflows/ci-cd.yaml).",
        "A Dockerfile that containerises the image classification web application.",
        "AWS S3 configured as the DVC remote, an AWS EC2 instance, and the Dockerised application deployed on it.",
        f"Verification of the deployment through the public IP of the EC2 instance ({APP_URL}).",
    ]),

    (H1, "2. Architecture and Tools"),
    (P, "End-to-end workflow:"),
    (CODE, "Developer laptop                     AWS S3 (DVC remote)\n"
           "  dvc repro  ── trains model ──►  dvc push  ──►  s3://mlops-imgcls-23070126144/dvcstore\n"
           "  git push   ──►  GitHub repository\n"
           "                     │\n"
           "                     ▼\n"
           "  GitHub Actions: checkout ─► dvc pull (model from S3) ─► docker build ─► smoke test\n"
           "                  ─► push image to ghcr.io ─► SSH to EC2 ─► docker pull + docker run -p 80:8080\n"
           "                     │\n"
           "                     ▼\n"
           "  AWS EC2 t2.micro (Ubuntu, Docker)  ──►  users open http://15.252.12.60/"),
    (TABLE, [
        ["Component", "Technology", "Purpose"],
        ["ML pipeline", "Python 3, TensorFlow 2.21 / Keras 3, OpenCV, scikit-learn", "Data processing, MobileNetV2 training and evaluation"],
        ["Pipeline orchestration and versioning", "DVC 3.67 (dvc.yaml, params.yaml, dvc.lock)", "Reproducible stages, dependency tracking, metrics"],
        ["Remote storage", "AWS S3 (ap-south-1)", "DVC remote holding the trained model, features and metrics"],
        ["Web application", "Flask + Gunicorn", "Upload page, JSON prediction API and health check"],
        ["Containerisation", "Docker (python:3.11-slim, tensorflow-cpu)", "Portable runtime image for the app"],
        ["CI/CD", "GitHub Actions + GitHub Container Registry", "Build, test, publish and deploy on every push"],
        ["Hosting", "AWS EC2 t2.micro, Ubuntu, free tier", "Runs the container on port 80 with a public IP"],
    ]),

    (H1, "3. Project Structure"),
    (CODE, "MLOps_ImageClassification_23070126144/\n"
           "├── .github/workflows/ci-cd.yaml   GitHub Actions CI/CD workflow\n"
           "├── .dvc/config                    DVC remote configuration (S3)\n"
           "├── src/                           pipeline stages\n"
           "│   ├── data_collection.py         downloads the Kaggle dataset\n"
           "│   ├── data_processing.py         resizes images, 70/20/10 train/validation/test split\n"
           "│   ├── feature_selection.py       MobileNetV2 feature extractor (ImageNet weights)\n"
           "│   ├── model_training.py          classification head training\n"
           "│   ├── model_evaluation.py        accuracy, precision, recall, F1, confusion matrix\n"
           "│   ├── config.py, utils.py        shared helpers\n"
           "├── app.py                         Flask web application\n"
           "├── Dockerfile, .dockerignore      container definition\n"
           "├── dvc.yaml, dvc.lock, params.yaml\n"
           "├── requirements.txt               pipeline dependencies\n"
           "├── requirements-app.txt           runtime dependencies of the container\n"
           "├── samples/                       one test image per class (used by CI smoke test)\n"
           "├── screenshots/                   screenshots used in this report\n"
           "└── report/                        this report and its generator"),

    (H1, "4. Step 1 - DVC Pipeline (dvc.yaml)"),
    (P, "All hyper-parameters (image size 224x224, batch size 32, 15 epochs, Adam with learning rate 1e-4, dropout 0.3, "
        "256 dense units, split ratios, augmentation settings and seed) live in params.yaml. dvc.yaml declares every stage "
        "with its command, code and data dependencies, the parameters it reads and the outputs it produces, so DVC re-runs "
        "only the stages whose inputs changed."),
    (TABLE, [
        ["Stage", "Command", "Main dependencies", "Outputs"],
        ["data_collection", "python -m src.data_collection", "params: dataset_url, dataset_path", "data/raw"],
        ["data_processing", "python -m src.data_processing", "data/raw, image_size, splits", "data/processed, artifacts/processed"],
        ["feature_selection", "python -m src.feature_selection", "data/processed, model_name", "artifacts/features"],
        ["model_training", "python -m src.model_training", "data/processed, artifacts/features, epochs, lr", "artifacts/model"],
        ["model_evaluation", "python -m src.model_evaluation", "data/processed, artifacts/model", "metrics.json, confusion_matrix.png"],
    ]),
    (P, "Commands used:"),
    (CODE, "git init -b main                      # DVC tracks pipeline state in Git\n"
           "python -m venv .venv && .venv\\Scripts\\activate\n"
           "pip install -r requirements.txt\n"
           "dvc repro                             # runs all five stages and writes dvc.lock\n"
           "dvc dag                               # shows the stage graph\n"
           "dvc metrics show                      # prints evaluation metrics"),
    (IMG, "04_dvc_dag.png", "DVC pipeline DAG (dvc dag) and dvc status after the run"),
    (IMG, "05_dvc_repro.png", "dvc repro executing all five stages, including 15 training epochs"),
    (P, "The Intel dataset used (3,000 images, 6 classes) is split into 2,100 training, 600 validation and 300 test images. "
        "Frozen MobileNetV2 features with a 256-unit dense head reach about 91% validation accuracy within 5 epochs."),
    (IMG, "20_training_history.png", "Training and validation curves produced by the model_training stage"),
    (IMG, "19_confusion_matrix.png", "Confusion matrix on the 300 held-out test images (model_evaluation stage)"),

    (H1, "5. Step 2 - AWS S3 as the DVC Remote"),
    (NUM, [
        "Created IAM user mlops-dvc. with programmatic access only and the AmazonS3FullAccess policy, then created an access key and stored it locally with aws configure.",
        "Created the S3 bucket mlops-imgcls-23070126144 in ap-south-1 (Mumbai) with Block Public Access enabled.",
        "Configured the bucket as the default DVC remote and pushed the pipeline artifacts.",
    ]),
    (CODE, "pip install \"dvc[s3]\"\n"
           "dvc remote add -d storage s3://mlops-imgcls-23070126144/dvcstore\n"
           "dvc remote modify storage region ap-south-1\n"
           "dvc push feature_selection model_training model_evaluation\n"
           "git add .dvc/config dvc.lock && git commit -m \"Configure S3 as DVC remote storage\""),
    (IMG, "02_aws_iam_user.png", "IAM user mlops-dvc. used by DVC and GitHub Actions"),
    (IMG, "01_aws_s3_bucket.png", "S3 bucket mlops-imgcls-23070126144 in ap-south-1"),
    (IMG, "01b_aws_s3_objects.png", "DVC content-addressed objects stored in the S3 bucket"),
    (IMG, "06_dvc_metrics_push_s3.png", "dvc metrics show, dvc push to the S3 remote and the resulting S3 objects"),
    (P, "Only the model, feature extractor and metrics stages are pushed (9 objects, 35 MB). The raw and processed images are "
        "reproducible from Kaggle by the data_collection stage, so leaving them out keeps the bucket well within the free "
        "tier's 2,000 PUT requests per month. A fresh clone running dvc pull model_training model_evaluation restores the "
        "model exactly as recorded in dvc.lock, which is what the CI pipeline does."),

    (H1, "6. Step 3 - Flask Web Application"),
    (P, "app.py loads the trained model (artifacts/model/mobilenetv2_intel_classifier.keras) once at start-up, resizes "
        "uploaded images to 224x224 RGB, rescales them to [0, 1] exactly like the training generators, and returns the "
        "predicted class with per-class probabilities. Endpoints:"),
    (BUL, [
        "GET / - HTML page to upload an image and view the prediction with probability bars.",
        "POST /predict - JSON API, e.g. curl -F file=@samples/forest.jpg http://15.252.12.60/predict",
        "GET /health - health check used by Docker HEALTHCHECK and the CI/CD pipeline.",
        "Invalid uploads return HTTP 400; uploads are limited to 10 MB.",
    ]),
    (IMG, "07_app_home_local.png", "Application home page running locally in Docker"),
    (IMG, "08_app_prediction_local.png", "Local prediction for a forest test image"),

    (H1, "7. Step 4 - Dockerfile and Docker Build"),
    (P, "The image is based on python:3.11-slim and installs only the runtime dependencies (tensorflow-cpu 2.21.0 and "
        "keras 3.15.1, the same versions used for training, plus Flask, Gunicorn, Pillow, NumPy and PyYAML). The "
        ".dockerignore whitelists only app.py, params.yaml, requirements-app.txt and the model file, so the build context "
        "is a few megabytes and never includes the dataset or virtual environment. The container runs as a non-root user, "
        "exposes port 8080 and defines a HEALTHCHECK."),
    code("Dockerfile"),
    (CODE, "dvc pull model_training\n"
           "docker build -t image-classifier:local .\n"
           "docker run -d -p 8080:8080 image-classifier:local\n"
           "curl http://localhost:8080/health\n"
           "curl -F file=@samples/forest.jpg http://localhost:8080/predict"),
    (IMG, "10_docker_build.png", "Docker build of the application image (2.2 GB, mostly TensorFlow)"),
    (IMG, "09_app_health_local.png", "Health endpoint of the local container"),

    (H1, "8. Step 5 - AWS EC2 Instance"),
    (TABLE, [
        ["Setting", "Value"],
        ["Name / region", "mlops-image-classifier, ap-south-1 (Mumbai)"],
        ["AMI / type", "Ubuntu Server LTS, t2.micro (free-tier eligible, 1 vCPU, 1 GB RAM)"],
        ["Storage", "20 GiB gp3 (within the 30 GB free tier)"],
        ["Key pair", "mlops-key (RSA, .pem) used by GitHub Actions for SSH"],
        ["Security group", "Inbound SSH 22 and HTTP 80 from 0.0.0.0/0"],
        ["Public IPv4", "15.252.12.60"],
    ]),
    (P, "The instance was bootstrapped with the following user-data script, which installs Docker and adds a 2 GB swap "
        "file so that TensorFlow fits comfortably in 1 GB of RAM:"),
    (CODE, "#!/bin/bash\n"
           "fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile\n"
           "echo '/swapfile none swap sw 0 0' >> /etc/fstab\n"
           "apt-get update -y && apt-get install -y docker.io\n"
           "systemctl enable --now docker\n"
           "usermod -aG docker ubuntu"),
    (IMG, "03_aws_ec2_instance.png", "EC2 t2.micro instance running in ap-south-1 with public IP 15.252.12.60"),
    (IMG, "03b_aws_security_group.png", "Security group inbound rules (SSH 22, HTTP 80)"),

    (H1, "9. Step 6 - GitHub Repository and Secrets"),
    (P, f"The code was pushed to the public repository {REPO}. The workflow uses four encrypted repository secrets, "
        "so no credential is stored in the code:"),
    (TABLE, [
        ["Secret", "Used for"],
        ["AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY", "dvc pull of the model from the S3 remote"],
        ["EC2_HOST", "Public IP of the EC2 instance"],
        ["EC2_SSH_KEY", "Private key of the EC2 key pair for SSH deployment"],
    ]),
    (P, "Publishing to GitHub Container Registry (ghcr.io) uses the built-in GITHUB_TOKEN, so no extra registry account is needed."),
    (IMG, "11_github_repo.png", "Public GitHub repository"),

    (H1, "10. Step 7 - GitHub Actions CI/CD (ci-cd.yaml)"),
    (P, "The workflow runs on every push and pull request to main, and can also be started manually. It has two jobs:"),
    (NUM, [
        "build-test: checks out the code, installs DVC, pulls the trained model and metrics from S3 (dvc pull), prints dvc dag "
        "and dvc metrics show, builds the Docker image, starts it and smoke-tests it by asserting that samples/forest.jpg is "
        "classified as forest, then pushes the image to ghcr.io tagged with the commit SHA and latest.",
        "deploy (only on main): connects to EC2 over SSH, logs in to ghcr.io with the short-lived workflow token, pulls the new "
        "image, replaces the running container (docker run -d --restart unless-stopped -p 80:8080), prunes old images and "
        "finally checks the /health endpoint on the EC2 public IP from the internet.",
    ]),
    code(".github/workflows/ci-cd.yaml"),
    (IMG, "12_github_actions_runs.png", "Successful CI/CD workflow runs"),
    (IMG, "13_github_actions_run.png", "Run #2: build-test (1m 56s) followed by deploy to AWS EC2 (2m 2s)"),
    (IMG, "14_ci_log_dvc_smoke_deploy.png", "Job logs: dvc pull from S3, DVC DAG and metrics, Docker build, smoke test, GHCR push and SSH deploy"),

    (H1, "11. Step 8 - Deployment Verification on EC2"),
    (P, f"The application is publicly available at {APP_URL}. The container started by the pipeline is reported as healthy "
        "by Docker, and predictions made over the internet match the local results."),
    (IMG, "18_ec2_docker_and_api.png", "Container running on EC2 and the public health and prediction API"),
    (IMG, "15_app_home_ec2.png", f"Application home page served from the EC2 public IP ({APP_URL})"),
    (IMG, "16_app_prediction_ec2.png", "Prediction for a glacier image on the deployed application"),
    (IMG, "17_app_health_ec2.png", "Health endpoint on the EC2 public IP"),

    (H1, "12. Results"),
    (TABLE, [
        ["Metric (test set, 300 images)", "Value"],
        ["Accuracy", f"{metrics['accuracy']:.2%}"],
        ["Precision (weighted)", f"{metrics['precision']:.2%}"],
        ["Recall (weighted)", f"{metrics['recall']:.2%}"],
        ["F1 score (weighted)", f"{metrics['f1_score']:.2%}"],
        ["Docker image size", "2.2 GB"],
        ["CI/CD duration (run #2)", "4 min 4 s (build-test 1m 56s, deploy 2m 2s)"],
    ]),

    (H1, "13. Challenges and Solutions"),
    (TABLE, [
        ["Challenge", "Solution"],
        ["The data_collection stage used the Kaggle CLI, which requires a Kaggle account and API token on every machine.",
         "Switched to Kaggle's public download endpoint using Python's urllib, removing the kaggle dependency, so dvc repro works on a fresh clone."],
        ["dvc repro failed with ModuleNotFoundError: matplotlib because DVC launched the system Python instead of the virtual environment.",
         "Activated the virtual environment (put .venv/Scripts first on PATH) before running dvc repro."],
        ["The project was initialised with DVC no_scm mode and dvc.lock was in .gitignore, so CI could not know which artifact versions to pull.",
         "Initialised Git, removed no_scm and committed dvc.lock together with dvc.yaml and params.yaml."],
        ["The local AWS CLI credentials were invalid (InvalidClientTokenId).",
         "Created a dedicated IAM user with only S3 access and a new access key, configured with aws configure."],
        ["TensorFlow images are large and a t2.micro has only 1 GB of RAM.",
         "Used tensorflow-cpu with a separate slim requirements-app.txt, a whitelist .dockerignore, a single Gunicorn worker and a 2 GB swap file. Images are built on GitHub runners, not on the EC2 instance."],
        ["A model saved with one Keras version may not load with another.",
         "Pinned tensorflow-cpu 2.21.0 and keras 3.15.1 in the container to match the training environment."],
        ["Staying strictly in the AWS free tier.",
         "t2.micro in standard CPU-credit mode, a single instance, 20 GB disk, no Elastic IP, and only 9 small objects pushed to S3 (well below the 2,000 free PUT requests)."],
        ["Container image names must be lowercase, but the GitHub repository name contains capitals.",
         "Lowercased the name in the workflow with the bash expansion ${GITHUB_REPOSITORY,,}."],
        ["Keeping credentials out of the code.",
         "Stored AWS keys, the EC2 host and the SSH key as encrypted GitHub secrets. GitHub masks them in logs, and the ghcr.io login on EC2 uses the short-lived workflow token followed by docker logout."],
        ["GitHub warned that Node.js 20 based actions are deprecated.",
         "Upgraded to actions/checkout@v5 and actions/setup-python@v6."],
    ]),

    (H1, "14. Conclusion"),
    (P, "The project demonstrates a complete MLOps workflow: a reproducible DVC pipeline trains a MobileNetV2 scene classifier "
        f"with {metrics['accuracy']:.1%} test accuracy, its artifacts are versioned in AWS S3, and every push to GitHub "
        "automatically pulls the model, builds and tests a Docker image, publishes it and redeploys it to an AWS EC2 "
        "instance, where it is publicly reachable."),

    (H1, "Appendix A - dvc.yaml"),
    code("dvc.yaml"),

    (H1, "GitHub Repository and Submission Links"),
    (P, f"GitHub repository (complete source code, public): {REPO}"),
    (P, f"Deployed application (AWS EC2 public IP): {APP_URL}"),
    (TABLE, [
        ["Item", "Link / location"],
        ["GitHub repository (public)", REPO],
        ["Deployed application (EC2 public IP)", APP_URL],
        ["Health check endpoint", f"{APP_URL}health"],
        ["Prediction API (POST, form field 'file')", f"{APP_URL}predict"],
        ["dvc.yaml", f"{REPO}/blob/main/dvc.yaml"],
        ["ci-cd.yaml", f"{REPO}/blob/main/.github/workflows/ci-cd.yaml"],
        ["Dockerfile", f"{REPO}/blob/main/Dockerfile"],
        ["Flask application (app.py)", f"{REPO}/blob/main/app.py"],
        ["GitHub Actions CI/CD runs", f"{REPO}/actions"],
        ["Successful CI/CD run #2", f"{REPO}/actions/runs/39301239608"],
        ["Screenshots", f"{REPO}/tree/main/screenshots"],
        ["Project report (PDF / Word)", f"{REPO}/tree/main/report"],
        ["DVC remote (AWS S3, ap-south-1)", "s3://mlops-imgcls-23070126144/dvcstore"],
        ["Docker image (GitHub Container Registry, private)", "ghcr.io/vedantsinghthakur21/mlops_imageclassification_23070126144"],
    ]),
]


def shade(element, color: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    element.append(shd)


URL = re.compile(r"(https?://[^\s)]*[^\s).,;])")


def add_text(paragraph, text: str, size: float | None = None) -> None:
    """Add text to a paragraph, turning http(s) URLs into clickable hyperlinks."""

    for i, piece in enumerate(URL.split(text)):
        if not piece:
            continue
        if i % 2 == 0:
            run = paragraph.add_run(piece)
            if size:
                run.font.size = Pt(size)
            continue
        link = OxmlElement("w:hyperlink")
        link.set(qn("r:id"), paragraph.part.relate_to(piece, RELATIONSHIP_TYPE.HYPERLINK, is_external=True))
        run = OxmlElement("w:r")
        props = OxmlElement("w:rPr")
        for tag, value in (("w:color", "1D4ED8"), ("w:u", "single"), ("w:sz", str(int((size or 11) * 2)))):
            element = OxmlElement(tag)
            element.set(qn("w:val"), value)
            props.append(element)
        run.append(props)
        text_element = OxmlElement("w:t")
        text_element.text = piece
        run.append(text_element)
        link.append(run)
        paragraph._p.append(link)


def linkify(text: str) -> str:
    return URL.sub(r"<a href='\1'>\1</a>", html.escape(text))


def build_docx(path: Path) -> None:
    global figure_count
    figure_count = 0
    doc = Document()
    for section in doc.sections:
        section.left_margin = section.right_margin = Inches(0.9)
        section.top_margin = section.bottom_margin = Inches(0.9)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(120)
    run = title.add_run("Deploying an Image Classification Application on AWS Using Docker, DVC, and CI/CD")
    run.bold, run.font.size = True, Pt(24)
    for text, size in (("MLOps Project Report", 16), ("", 11), ("Vedant Singh Thakur", 14), ("PRN: 23070126144", 12),
                       ("Symbiosis Institute of Technology, Pune", 12), ("October 2026", 12), ("", 11),
                       (f"GitHub: {REPO}", 10), (f"Application: {APP_URL}", 10)):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_text(p, text, size)
    doc.add_page_break()

    for kind, *rest in CONTENT:
        if kind == H1:
            doc.add_heading(rest[0], level=1)
        elif kind == P:
            add_text(doc.add_paragraph(), rest[0])
        elif kind in (BUL, NUM):
            for item in rest[0]:
                add_text(doc.add_paragraph(style="List Bullet" if kind == BUL else "List Number"), item)
        elif kind == CODE:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(8)
            shade(p._p.get_or_add_pPr(), "F3F4F6")
            lines = rest[0].split("\n")
            for i, line in enumerate(lines):
                r = p.add_run(line)
                r.font.name, r.font.size = "Consolas", Pt(8.5)
                r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
                if i < len(lines) - 1:
                    r.add_break()
        elif kind == IMG:
            file, caption = rest
            if not (SHOTS / file).exists():
                continue
            figure_count += 1
            doc.add_picture(str(SHOTS / file), width=Inches(6.4))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cap.add_run(f"Figure {figure_count}: {caption}")
            r.italic, r.font.size, r.font.color.rgb = True, Pt(9.5), RGBColor(0x4B, 0x55, 0x63)
        elif kind == TABLE:
            rows = rest[0]
            table = doc.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Table Grid"
            for r_i, row in enumerate(rows):
                for c_i, value in enumerate(row):
                    cell = table.cell(r_i, c_i)
                    add_text(cell.paragraphs[0], value, 9.5)
                    for r in cell.paragraphs[0].runs:
                        r.bold = r_i == 0
                    if r_i == 0:
                        shade(cell._tc.get_or_add_tcPr(), "DBEAFE")
            doc.add_paragraph()
    doc.save(path)


def build_html() -> str:
    count = 0
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><style>",
        "@page{size:A4;margin:18mm} body{font-family:Calibri,Segoe UI,sans-serif;font-size:11pt;color:#111827;line-height:1.4}",
        "h1{font-size:16pt;color:#1e3a8a;border-bottom:1px solid #cbd5e1;padding-bottom:3px;margin-top:22px;break-after:avoid}",
        "pre{background:#f3f4f6;padding:8px;font-size:8pt;white-space:pre-wrap;font-family:Consolas,monospace;break-inside:avoid}",
        "figure{margin:12px 0;text-align:center;break-inside:avoid} img{max-width:100%;max-height:200mm;border:1px solid #e5e7eb}",
        "figcaption{font-style:italic;font-size:9.5pt;color:#4b5563} table{border-collapse:collapse;width:100%;margin:8px 0;font-size:9.5pt}",
        "td,th{border:1px solid #9ca3af;padding:4px 6px;vertical-align:top;text-align:left} th{background:#dbeafe}",
        ".title{text-align:center;padding-top:60mm;break-after:page} .title h2{font-size:24pt;margin-bottom:6px}",
        "</style></head><body>",
        "<div class='title'><h2>Deploying an Image Classification Application on AWS Using Docker, DVC, and CI/CD</h2>",
        "<p style='font-size:16pt'>MLOps Project Report</p><p style='font-size:14pt'>Vedant Singh Thakur<br>PRN: 23070126144<br>"
        "Symbiosis Institute of Technology, Pune<br>October 2026</p>",
        f"<p>GitHub: {linkify(REPO)}<br>Application: {linkify(APP_URL)}</p></div>",
    ]
    for kind, *rest in CONTENT:
        if kind == H1:
            parts.append(f"<h1>{html.escape(rest[0])}</h1>")
        elif kind == P:
            parts.append(f"<p>{linkify(rest[0])}</p>")
        elif kind in (BUL, NUM):
            tag = "ul" if kind == BUL else "ol"
            parts.append(f"<{tag}>" + "".join(f"<li>{linkify(i)}</li>" for i in rest[0]) + f"</{tag}>")
        elif kind == CODE:
            parts.append(f"<pre>{html.escape(rest[0])}</pre>")
        elif kind == IMG and (SHOTS / rest[0]).exists():
            count += 1
            parts.append(f"<figure><img src='{(SHOTS / rest[0]).as_uri()}'><figcaption>Figure {count}: {html.escape(rest[1])}</figcaption></figure>")
        elif kind == TABLE:
            head, *body = rest[0]
            parts.append("<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in head) + "</tr>"
                         + "".join("<tr>" + "".join(f"<td>{linkify(c)}</td>" for c in row) + "</tr>" for row in body) + "</table>")
    parts.append("</body></html>")
    return "".join(parts)


if __name__ == "__main__":
    build_docx(OUT / f"{NAME}.docx")
    html_path = OUT / f"{NAME}.html"
    html_path.write_text(build_html(), encoding="utf-8")
    if EDGE.exists():
        subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={Path.home() / 'AppData/Local/Temp/edge-report'}",
                        f"--print-to-pdf={OUT / f'{NAME}.pdf'}", html_path.as_uri()], check=False, timeout=120)
    html_path.unlink(missing_ok=True)
    print(f"Wrote {OUT / f'{NAME}.docx'} ({figure_count} figures) and {OUT / f'{NAME}.pdf'}")
