"""Generate the experiment report PDF for PRML Assignment 2."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import os

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
OUTPUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report2.pdf')

def build_report():
    doc = SimpleDocTemplate(
        OUTPUT_PDF, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=18, spaceAfter=6)
    author_style = ParagraphStyle('Author', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER, spaceAfter=2)
    h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=15, spaceBefore=16, spaceAfter=8)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13, spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
    caption = ParagraphStyle('Caption', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, spaceAfter=10, textColor=HexColor('#555555'))

    story = []

    # ===== Title =====
    story.append(Paragraph('PRML Assignment 2: Classification Experiment Report', title_style))
    story.append(Paragraph('Wang Hongyu (22371297)', author_style))
    story.append(Paragraph('2026 Spring', author_style))
    story.append(Spacer(1, 8*mm))

    # ===== Abstract =====
    story.append(Paragraph('Abstract', h1))
    story.append(Paragraph(
        'This report investigates the classification performance of several machine learning models on a 3D dataset generated '
        'using a modified "make-moons" algorithm. The study compares Decision Trees, AdaBoost (with Decision Trees), '
        'and Support Vector Machines (SVM) across three distinct kernel functions: Linear, Polynomial, and RBF. '
        'Experiments demonstrate that for highly nonlinear 3D manifold data, kernel-based methods and ensemble learning '
        'significantly outperform linear boundaries. AdaBoost achieved the highest accuracy of 0.990, followed closely by '
        'RBF SVM at 0.988, whereas Linear SVM remained inadequate with an accuracy of 0.698.',
        body
    ))

    # ===== 1. Introduction =====
    story.append(Paragraph('1. Introduction', h1))
    story.append(Paragraph(
        'Classification is a core task in pattern recognition, aiming to assign input vectors into distinct categories. '
        'In realistic datasets, classes are often intertwined in high-dimensional space, requiring non-linear decision boundaries.',
        body
    ))
    story.append(Paragraph(
        'In this assignment, we work with a 3D dataset (1000 training points and 500 test points) where two classes (C0 and C1) '
        'are generated sequentially to form a curvilinear structure. The task involves:',
        body
    ))
    story.append(Paragraph(
        '<b>(1)</b> Evaluate the baseline classification performance of Decision Trees.<br/>'
        '<b>(2)</b> Analyze the performance boost provided by ensemble methods like AdaBoost.<br/>'
        '<b>(3)</b> Investigate the impact of different SVM kernels on capture-ability of the decision surface.',
        body
    ))

    # ===== 2. Methodology =====
    story.append(Paragraph('2. Methodology', h1))

    story.append(Paragraph('2.1 Classification Models', h2))
    story.append(Paragraph(
        '<b>Decision Trees (DT):</b> Non-parametric learning method that creates a model predicting values by learning '
        'simple decision rules inferred from features.<br/><br/>'
        '<b>AdaBoost:</b> An ensemble method that combines multiple "weak" learners (shallow decision trees) into a strong '
        'learner by iteratively focusing on misclassified samples.<br/><br/>'
        '<b>Support Vector Machines (SVM):</b> Optimization of the margin between classes. We test three kernels:<br/>'
        '  - <i>Linear:</i> Standard dot-product. Ideal for linearly separable data.<br/>'
        '  - <i>Polynomial (d=3):</i> Standard polynomial expansion.<br/>'
        '  - <i>RBF (Gaussian):</i> Measures local similarity via radial basis functions.',
        body
    ))

    # ===== 3. Experimental Studies =====
    story.append(PageBreak())
    story.append(Paragraph('3. Experimental Studies', h1))

    story.append(Paragraph('3.1 Data Distribution', h2))
    story.append(Paragraph(
        'The 3D Make-Moons data is generated with a noise level of 0.2. As shown in Figure 1, the two classes '
        'are highly non-linear and exhibit a manifold-like structure in 3D space.',
        body
    ))
    img_path = os.path.join(FIGURE_DIR, 'hw2_data_distribution.png')
    story.append(Image(img_path, width=14*cm, height=9.8*cm))
    story.append(Paragraph('Figure 1: 3D Make-Moons training data distribution showing the non-linear manifold.', caption))

    story.append(Paragraph('3.2 Quantitative Results', h2))
    story.append(Paragraph(
        'Table 1 summarizes the accuracy and F1-score for all models on the held-out test set (500 samples).',
        body
    ))

    # Table 1
    table_data = [
        ['Model', 'Accuracy', 'F1-Score', 'Grade'],
        ['Decision Tree', '0.9540', '0.9533', 'Good'],
        ['AdaBoost (DT)', '0.9900', '0.9900', 'Best'],
        ['SVM (Linear)', '0.6980', '0.7010', 'Poor'],
        ['SVM (Poly d=3)', '0.8800', '0.8661', 'Moderate'],
        ['SVM (RBF)', '0.9880', '0.9880', 'Excellent'],
    ]
    t = Table(table_data, colWidths=[4*cm, 3*cm, 3*cm, 3*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2196F3')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#F5F5F5'), HexColor('#FFFFFF')]),
    ]))
    story.append(t)
    story.append(Paragraph('Table 1: Classification results comparison on 3D test set.', caption))

    img_path = os.path.join(FIGURE_DIR, 'hw2_performance_comparison.png')
    story.append(Image(img_path, width=14*cm, height=7.5*cm))
    story.append(Paragraph('Figure 2: Performance metrics comparison across all five models.', caption))

    # ===== 4. Conclusions =====
    story.append(Paragraph('4. Conclusions', h1))
    story.append(Paragraph(
        'The experiments conclude that:<br/><br/>'
        '<b>(1)</b> Linear SVM is unsuitable for Moons-pattern data, as no simple hyper-plane can separate the classes.<br/><br/>'
        '<b>(2)</b> SVM with RBF kernel and the AdaBoost ensemble successfully captured the complex data structure.<br/><br/>'
        '<b>(3)</b> Ensemble learning (AdaBoost) proved most effective, reaching 99% accuracy through iterative boundary refinement.',
        body
    ))

    # ===== References =====
    story.append(Paragraph('References', h1))
    story.append(Paragraph(
        '[1] C.M. Bishop, <i>Pattern Recognition and Machine Learning</i>, Springer, 2006.<br/>'
        '[2] F. Pedregosa et al., <i>Scikit-learn: Machine Learning in Python</i>, JMLR, 2011.',
        body
    ))

    doc.build(story)
    print(f'Report saved to: {OUTPUT_PDF}')

if __name__ == '__main__':
    build_report()
