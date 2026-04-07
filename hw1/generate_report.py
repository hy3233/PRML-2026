"""Generate the experiment report PDF for PRML Assignment 1."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import os

FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
OUTPUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report.pdf')

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
    story.append(Paragraph('PRML Assignment 1: Regression Experiment Report', title_style))
    story.append(Paragraph('Wang Hongyu (22371297)', author_style))
    story.append(Paragraph('2026 Spring', author_style))
    story.append(Spacer(1, 8*mm))

    # ===== Abstract =====
    story.append(Paragraph('Abstract', h1))
    story.append(Paragraph(
        'This report presents a comprehensive study on regression methods applied to a given 2D dataset. '
        'We first implement three linear regression approaches: Ordinary Least Squares (OLS), Gradient Descent (GD), '
        "and Newton's Method. Since the data exhibits clear nonlinear patterns, we further explore polynomial regression "
        'with different degrees and Gaussian basis function regression. L2 regularization (Ridge Regression) is also investigated '
        'to address overfitting in high-degree polynomial models. The Gaussian RBF model achieves the best test MSE of 0.2868, '
        'significantly outperforming linear models (test MSE ~ 0.595).',
        body
    ))

    # ===== 1. Introduction =====
    story.append(Paragraph('1. Introduction', h1))
    story.append(Paragraph(
        'Regression is a fundamental task in machine learning, aiming to learn a mapping from input features to continuous '
        'output values. In this assignment, we are given a 2D dataset (x, y) with 100 training samples and 100 test samples, '
        'where x is in [0, 10]. The task involves two parts:',
        body
    ))
    story.append(Paragraph(
        '<b>(1)</b> Fit a linear model y = w0 + w1*x using three optimization methods (OLS, GD, Newton), and compare '
        'their training and test errors.<br/>'
        '<b>(2)</b> Since the data is nonlinear, explore more appropriate models and provide model selection rationale, '
        'experimental results, and analysis.',
        body
    ))

    # ===== 2. Methodology =====
    story.append(Paragraph('2. Methodology', h1))

    story.append(Paragraph('2.1 Linear Regression', h2))
    story.append(Paragraph(
        'For a linear model y = w0 + w1*x, we define the design matrix <b>X</b> = [1, x] and solve for the weight vector '
        '<b>w</b> = [w0, w1]<super>T</super> that minimizes MSE = (1/N) * sum( (y_i - x_i<super>T</super> w)<super>2</super> ).',
        body
    ))
    story.append(Paragraph(
        '<b>OLS (Ordinary Least Squares):</b> Closed-form solution <b>w</b> = (<b>X</b><super>T</super><b>X</b>)'
        '<super>-1</super> <b>X</b><super>T</super> <b>y</b>. '
        'Provides the exact optimal solution in one step.<br/><br/>'
        '<b>Gradient Descent (GD):</b> Iterative optimization with update rule '
        '<b>w</b> &lt;- <b>w</b> - lr * grad(L), where '
        'grad(L) = (2/N) <b>X</b><super>T</super>(<b>X</b>w - y). We use learning rate lr = 0.001 for 5000 iterations.<br/><br/>'
        "<b>Newton's Method:</b> Second-order optimization using the Hessian matrix "
        '<b>H</b> = (2/N) <b>X</b><super>T</super><b>X</b>. '
        'Update rule: <b>w</b> &lt;- <b>w</b> - <b>H</b><super>-1</super> grad(L). '
        'Converges in just 1-2 iterations for linear problems.',
        body
    ))

    story.append(Paragraph('2.2 Nonlinear Models', h2))
    story.append(Paragraph(
        '<b>Polynomial Regression:</b> We extend the feature space using polynomial basis functions '
        'phi(x) = [1, x, x<super>2</super>, ..., x<super>M</super>]. '
        'The model becomes y = sum(w_j * x<super>j</super>). Higher degree M captures more complex patterns '
        'but risks overfitting. We normalize x to zero-mean unit-variance before computing polynomial features '
        'to prevent numerical overflow.<br/><br/>'
        '<b>Gaussian Basis Functions:</b> We use phi_j(x) = exp(-(x - mu_j)<super>2</super> / (2*sigma<super>2</super>)) '
        'with K=15 basis functions uniformly '
        'distributed across [0, 10] with sigma = 0.5.<br/><br/>'
        '<b>L2 Regularization (Ridge Regression):</b> To combat overfitting in high-degree polynomials, we add '
        'a penalty term: L = MSE + lambda * ||w||<super>2</super>. '
        'The solution becomes <b>w</b> = (Phi<super>T</super> Phi + lambda * <b>I</b>)<super>-1</super> '
        'Phi<super>T</super> <b>y</b>.',
        body
    ))

    # ===== 3. Experimental Studies =====
    story.append(PageBreak())
    story.append(Paragraph('3. Experimental Studies', h1))

    story.append(Paragraph('3.1 Data Overview', h2))
    story.append(Paragraph(
        'The dataset consists of 100 training points and 100 test points. As shown in Figure 1, '
        'the data exhibits a clear nonlinear oscillation pattern with noise, making linear fitting inherently limited.',
        body
    ))
    img_path = os.path.join(FIGURE_DIR, 'data_overview.png')
    story.append(Image(img_path, width=14*cm, height=8.4*cm))
    story.append(Paragraph('Figure 1: Training and test data scatter plot. The data shows nonlinear periodic patterns.', caption))

    story.append(Paragraph('3.2 Linear Regression Results', h2))
    story.append(Paragraph(
        'Table 1 summarizes the results of three linear regression methods. All three converge to nearly '
        'identical solutions, confirming correctness. The test MSE (~0.595) is high, indicating poor linear fit.',
        body
    ))

    # Table 1
    table_data = [
        ['Method', 'w0', 'w1', 'Train MSE', 'Test MSE'],
        ['OLS', '-0.6487', '0.1089', '0.6134', '0.5950'],
        ['GD', '-0.6440', '0.1080', '0.6142', '0.5934'],
        ['Newton', '-0.6487', '0.1089', '0.6134', '0.5950'],
    ]
    t = Table(table_data, colWidths=[3*cm, 2.2*cm, 2.2*cm, 2.5*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2196F3')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#F5F5F5'), HexColor('#FFFFFF')]),
    ]))
    story.append(t)
    story.append(Paragraph('Table 1: Linear regression results. All methods converge to similar solutions.', caption))

    img_path = os.path.join(FIGURE_DIR, 'linear_comparison.png')
    story.append(Image(img_path, width=14*cm, height=8.4*cm))
    story.append(Paragraph('Figure 2: Comparison of three linear fitting methods. All produce nearly identical fits.', caption))

    img_path = os.path.join(FIGURE_DIR, 'gd_convergence.png')
    story.append(Image(img_path, width=12*cm, height=7.2*cm))
    story.append(Paragraph('Figure 3: Gradient Descent convergence curve showing MSE decreasing over 5000 iterations.', caption))

    img_path = os.path.join(FIGURE_DIR, 'newton_convergence.png')
    story.append(Image(img_path, width=12*cm, height=7.2*cm))
    story.append(Paragraph("Figure 4: Newton's Method convergence - reaches optimum in just 2 iterations.", caption))

    # ===== 3.3 Nonlinear Results =====
    story.append(PageBreak())
    story.append(Paragraph('3.3 Polynomial Regression Results', h2))
    story.append(Paragraph(
        'Since the linear model is clearly insufficient, we apply polynomial regression with varying degrees. '
        'We normalize x before constructing polynomial features to ensure numerical stability for high degrees.',
        body
    ))

    img_path = os.path.join(FIGURE_DIR, 'nonlinear_poly.png')
    story.append(Image(img_path, width=16*cm, height=10*cm))
    story.append(Paragraph('Figure 5: Polynomial fits with degrees M = 3, 5, 7, 9, 12, 15.', caption))

    # Table 2
    table_data2 = [
        ['Degree M', 'Train MSE', 'Test MSE'],
        ['3', '0.5653', '0.5368'],
        ['5', '0.5252', '0.5151'],
        ['7', '0.4651', '0.4631'],
        ['9', '0.3541', '0.3875'],
        ['12', '0.2968', '0.3425'],
        ['15', '0.2736', '0.3488'],
    ]
    t2 = Table(table_data2, colWidths=[3.5*cm, 3.5*cm, 3.5*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#009688')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#FFFFFF')),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#F5F5F5'), HexColor('#FFFFFF')]),
    ]))
    story.append(t2)
    story.append(Paragraph('Table 2: Polynomial regression errors for different degrees.', caption))

    img_path = os.path.join(FIGURE_DIR, 'poly_error_curve.png')
    story.append(Image(img_path, width=13*cm, height=7.8*cm))
    story.append(Paragraph(
        'Figure 6: Train/Test MSE vs polynomial degree. The gap between curves shows the bias-variance tradeoff.',
        caption
    ))

    story.append(Paragraph('3.4 L2 Regularization (Ridge Regression)', h2))
    story.append(Paragraph(
        'For high-degree polynomials (M=15), we apply L2 regularization with different lambda values. '
        'As lambda increases, the model becomes smoother (lower variance but higher bias). '
        'lambda = 0.01 provides a good balance with Test MSE = 0.3807.',
        body
    ))

    img_path = os.path.join(FIGURE_DIR, 'regularization.png')
    story.append(Image(img_path, width=16*cm, height=10*cm))
    story.append(Paragraph(
        'Figure 7: Effect of L2 regularization on degree-15 polynomial. '
        'Larger lambda progressively smooths the fit.',
        caption
    ))

    story.append(PageBreak())
    story.append(Paragraph('3.5 Gaussian Basis Function Regression', h2))
    story.append(Paragraph(
        'We use 15 Gaussian basis functions uniformly distributed in [0, 10] with sigma = 0.5, '
        'plus L2 regularization (lambda = 0.01). This achieves the best overall performance '
        'with Train MSE = 0.1858 and Test MSE = 0.2868.',
        body
    ))

    img_path = os.path.join(FIGURE_DIR, 'nonlinear_gauss.png')
    story.append(Image(img_path, width=14*cm, height=8.4*cm))
    story.append(Paragraph('Figure 8: Gaussian RBF regression achieves the best fit among all methods.', caption))

    story.append(Paragraph('3.6 Overall Comparison', h2))
    img_path = os.path.join(FIGURE_DIR, 'error_comparison.png')
    story.append(Image(img_path, width=15*cm, height=7.5*cm))
    story.append(Paragraph('Figure 9: Error comparison across all methods. Gaussian RBF achieves the lowest test error.', caption))

    # ===== 4. Conclusions =====
    story.append(Paragraph('4. Conclusions', h1))
    story.append(Paragraph(
        'This experiment demonstrates several key findings:<br/><br/>'
        '<b>(1)</b> All three linear optimization methods (OLS, GD, Newton) converge to the same solution, '
        'with Newton being the fastest (2 iterations). However, linear models are inadequate for this data '
        '(test MSE ~ 0.595).<br/><br/>'
        '<b>(2)</b> Polynomial regression significantly improves fit quality. Increasing degree reduces training error, '
        'but the test error gap demonstrates the classic bias-variance tradeoff. Degree M=9 offers a good balance '
        'without regularization.<br/><br/>'
        '<b>(3)</b> L2 regularization effectively controls overfitting in high-degree polynomials, allowing '
        'smoother fits with competitive test performance.<br/><br/>'
        '<b>(4)</b> Gaussian basis function regression achieves the best overall result (test MSE = 0.2868), '
        'suggesting that localized basis functions are well-suited for capturing the periodic oscillatory pattern '
        'in the data.',
        body
    ))

    # ===== References =====
    story.append(Paragraph('References', h1))
    story.append(Paragraph(
        '[1] C.M. Bishop, <i>Pattern Recognition and Machine Learning</i>, Springer, 2006.<br/>'
        '[2] T. Hastie, R. Tibshirani, J. Friedman, <i>The Elements of Statistical Learning</i>, 2nd ed., Springer, 2009.',
        body
    ))

    doc.build(story)
    print(f'Report saved to: {OUTPUT_PDF}')


if __name__ == '__main__':
    build_report()
