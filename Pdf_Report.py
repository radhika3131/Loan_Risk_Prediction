from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Create PDF report
def create_pdf_report():
    doc = SimpleDocTemplate("ML_Model_Report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Machine Learning Model Report", styles['Title']))

    # Data Visualizations
    elements.append(Paragraph("Data Visualizations", styles['Heading2']))
    elements.append(Image("dist_risk_flag.png", width=400, height=200))
    elements.append(Spacer(1, 12))
    elements.append(Image("corr_matrix.png", width=400, height=200))
    elements.append(Spacer(1, 12))
    elements.append(Image("pairplot.png", width=400, height=200))
    elements.append(Spacer(1, 12))

    # Data Exploration Insights
    elements.append(Paragraph("Data Exploration Insights", styles['Heading2']))
    elements.append(Paragraph("The data exploration revealed the following insights:", styles['BodyText']))
    elements.append(Paragraph("- The target variable 'Risk_Flag' is imbalanced with more instances of low risk.", styles['BodyText']))
    elements.append(Paragraph("- The correlation matrix indicates some correlation between income, age, and experience with the risk flag.", styles['BodyText']))
    elements.append(Spacer(1, 12))

    # Model Performance
    elements.append(Paragraph("Model Performance", styles['Heading2']))
    elements.append(Paragraph("The confusion matrix and classification report of the model are as follows:", styles['BodyText']))
    elements.append(Paragraph(f"Confusion Matrix:\n{conf_matrix}", styles['BodyText']))
    elements.append(Paragraph(f"Classification Report:\n{class_report}", styles['BodyText']))
    elements.append(Spacer(1, 12))

    # Feature Importances
    elements.append(Paragraph("Feature Importances", styles['Heading2']))
    elements.append(Image("feature_importances.png", width=400, height=200))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("The top features contributing to the risk prediction are:", styles['BodyText']))
    for i in range(min(10, len(importance_df))):
        elements.append(Paragraph(f"- {importance_df.iloc[i]['Feature']}: {importance_df.iloc[i]['Importance']:.4f}", styles['BodyText']))

    # Build PDF
    doc.build(elements)

create_pdf_report()