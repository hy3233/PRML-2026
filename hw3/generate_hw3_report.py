"""Generate the experiment report PDF for PRML Assignment 3."""

from __future__ import annotations

import json
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
RESULT_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_PDF = os.path.join(BASE_DIR, "report3.pdf")


def metric_table_flowable(metrics: pd.DataFrame) -> Table:
    rows = [["Model", "MAE", "RMSE", "R2"]]
    for name, row in metrics.iterrows():
        rows.append([name, f"{row['MAE']:.3f}", f"{row['RMSE']:.3f}", f"{row['R2']:.3f}"])

    table = Table(rows, colWidths=[4.4 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F6B9A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#F5F7FA"), colors.white]),
            ]
        )
    )
    return table


def image(path: str, width_cm: float, height_cm: float) -> Image:
    return Image(os.path.join(FIGURE_DIR, path), width=width_cm * cm, height=height_cm * cm)


def build_report() -> None:
    metrics = pd.read_csv(os.path.join(RESULT_DIR, "metrics.csv"), index_col=0)
    with open(os.path.join(RESULT_DIR, "summary.json"), "r", encoding="utf-8") as fh:
        summary = json.load(fh)

    best_model = summary["best_model"]
    best = summary["best_metrics"]

    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=2.4 * cm,
        rightMargin=2.4 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("CustomTitle", parent=styles["Title"], fontSize=18, spaceAfter=6)
    author = ParagraphStyle("Author", parent=styles["Normal"], fontSize=12, alignment=TA_CENTER, spaceAfter=2)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, spaceBefore=14, spaceAfter=8)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10.4,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=8.8,
        alignment=TA_CENTER,
        spaceAfter=9,
        textColor=HexColor("#555555"),
    )

    story = []
    story.append(Paragraph("PRML Assignment 3: Air Quality Forecasting Report", title))
    story.append(Paragraph("Wang Hongyu (22371297)", author))
    story.append(Paragraph("2026 Spring", author))
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("Abstract", h1))
    story.append(
        Paragraph(
            "This assignment studies one-hour-ahead PM2.5 forecasting with the Beijing Air Quality dataset. "
            "The raw data contains hourly pollution, meteorological variables, and wind direction from 2010 to 2014. "
            f"The forecasting problem is converted into supervised learning by using the previous {summary['lookback_hours']} "
            "hours of multivariate observations to predict the next PM2.5 value. A persistence baseline, Ridge regression, "
            "Random Forest, Gradient Boosting, a multilayer perceptron, and a PyTorch LSTM are evaluated on a chronological hold-out set. "
            f"The best model is {best_model}, with RMSE={best['RMSE']:.3f}, MAE={best['MAE']:.3f}, and R2={best['R2']:.3f}.",
            body,
        )
    )

    story.append(Paragraph("1. Dataset and Task Definition", h1))
    story.append(
        Paragraph(
            "The dataset records hourly weather and pollution conditions. Its variables include PM2.5 concentration, dew point, "
            "temperature, pressure, wind direction, wind speed, cumulative snow hours, and cumulative rain hours. The target is "
            "the pollution concentration in the next hour. Because future values must not leak into training, the data is split "
            "chronologically: the first 80% of rows are used for training and the final 20% are reserved for testing.",
            body,
        )
    )
    story.append(image("hw3_data_overview.png", 15.2, 9.4))
    story.append(Paragraph("Figure 1: Exploratory views of PM2.5, hourly/monthly patterns, and wind direction counts.", caption))

    story.append(Paragraph("2. Feature Engineering", h1))
    story.append(
        Paragraph(
            f"For each target hour, the previous {summary['lookback_hours']} rows are used as one lag window. "
            "Numeric variables are standardized using only the training period. Wind direction is one-hot encoded with categories "
            "learned from the training data. Traditional regressors receive the window as a flattened feature vector, while the "
            "LSTM receives the same data as an ordered 24-step sequence.",
            body,
        )
    )
    story.append(image("hw3_correlation.png", 12.0, 8.5))
    story.append(Paragraph("Figure 2: Correlation matrix of the numeric variables.", caption))

    story.append(PageBreak())
    story.append(Paragraph("3. Models", h1))
    story.append(
        Paragraph(
            "<b>Persistence baseline:</b> predicts that the next PM2.5 value equals the previous hour. "
            "<b>Ridge regression:</b> tests whether a regularized linear model captures useful lag structure. "
            "<b>Random Forest:</b> models nonlinear thresholds and interactions through decision-tree ensembles. "
            "<b>Gradient Boosting:</b> builds an additive nonlinear regressor optimized by sequential residual correction. "
            "<b>MLP:</b> uses a feed-forward neural network on the same lagged feature vector. "
            "<b>LSTM:</b> uses a recurrent neural network designed for sequence data and predicts from the final hidden state.",
            body,
        )
    )

    story.append(Paragraph("4. Results", h1))
    story.append(
        Paragraph(
            f"The supervised dataset contains {summary['train_samples']} training windows and {summary['test_samples']} test windows. "
            "The table below reports MAE, RMSE, and R2 on the chronological hold-out set. Lower MAE/RMSE and higher R2 indicate better forecasts.",
            body,
        )
    )
    story.append(metric_table_flowable(metrics))
    story.append(Paragraph("Table 1: Forecasting results on the temporal hold-out test set.", caption))
    story.append(image("hw3_metrics_comparison.png", 15.2, 5.0))
    story.append(Paragraph("Figure 3: Metric comparison across all forecasting models.", caption))

    story.append(image("hw3_prediction_series.png", 15.2, 5.8))
    story.append(Paragraph("Figure 4: First 240 test hours: actual PM2.5, best model forecast, and persistence baseline.", caption))

    story.append(PageBreak())
    story.append(image("hw3_prediction_scatter.png", 10.8, 10.0))
    story.append(Paragraph(f"Figure 5: Actual vs predicted PM2.5 for {best_model}.", caption))

    story.append(image("hw3_lstm_training.png", 12.0, 7.0))
    story.append(Paragraph("Figure 6: LSTM training and validation loss curves.", caption))

    story.append(image("hw3_feature_importance.png", 15.2, 5.6))
    story.append(
        Paragraph(
            "Figure 7: Random Forest lag-feature importance, aggregated by original variable and by hour-before-target.",
            caption,
        )
    )

    story.append(Paragraph("5. Discussion", h1))
    story.append(
        Paragraph(
            "The persistence baseline is strong because air pollution changes smoothly over adjacent hours. Ridge regression improves "
            "or degrades depending on how linear the local dynamics are, while tree ensembles can model nonlinear interactions among "
            "recent pollution, wind, temperature, and pressure. MLP and LSTM add neural-network baselines: the MLP treats the lag "
            "window as one vector, whereas the LSTM explicitly processes the observations as a sequence.",
            body,
        )
    )
    story.append(
        Paragraph(
            f"The best result, {best_model}, suggests that nonlinear temporal patterns are important for this forecasting task. "
            "The lag-importance plot also shows that the most recent pollution values dominate the prediction, which is consistent "
            "with the temporal autocorrelation expected in hourly air-quality data.",
            body,
        )
    )

    story.append(Paragraph("6. Conclusion", h1))
    story.append(
        Paragraph(
            "This experiment completes the multivariate air-quality forecasting pipeline: data loading, preprocessing, lag-window "
            "construction, model comparison, and quantitative evaluation. The chronological split keeps the evaluation realistic, "
            "and the included baseline helps distinguish genuine modeling gains from the natural persistence of PM2.5 values.",
            body,
        )
    )

    story.append(Paragraph("References", h1))
    story.append(
        Paragraph(
            "[1] C. M. Bishop, <i>Pattern Recognition and Machine Learning</i>, Springer, 2006.<br/>"
            "[2] F. Pedregosa et al., <i>Scikit-learn: Machine Learning in Python</i>, JMLR, 2011.<br/>"
            "[3] Kaggle, LSTM Datasets: Multivariate and Univariate Air Quality Data.",
            body,
        )
    )

    doc.build(story)
    print(f"Report saved to: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_report()
