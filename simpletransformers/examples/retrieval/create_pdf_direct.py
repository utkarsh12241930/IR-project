"""
Create PDF directly using reportlab
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
report_file = os.path.join(script_dir, "PROJECT_REPORT.md")
output_pdf = os.path.join(script_dir, "HANS_IR_Project_DS501.pdf")

def parse_markdown(md_text):
    """Parse markdown and return structured content."""
    lines = md_text.split('\n')
    content = []
    current_para = []
    in_code = False
    code_lines = []
    
    for line in lines:
        # Code blocks
        if line.strip().startswith('```'):
            if in_code:
                if code_lines:
                    content.append(('code', '\n'.join(code_lines)))
                    code_lines = []
                in_code = False
            else:
                if current_para:
                    content.append(('para', '\n'.join(current_para)))
                    current_para = []
                in_code = True
            continue
        
        if in_code:
            code_lines.append(line)
            continue
        
        # Headers
        if line.startswith('# '):
            if current_para:
                content.append(('para', '\n'.join(current_para)))
                current_para = []
            content.append(('h1', line[2:].strip()))
        elif line.startswith('## '):
            if current_para:
                content.append(('para', '\n'.join(current_para)))
                current_para = []
            content.append(('h2', line[3:].strip()))
        elif line.startswith('### '):
            if current_para:
                content.append(('para', '\n'.join(current_para)))
                current_para = []
            content.append(('h3', line[4:].strip()))
        elif line.startswith('#### '):
            if current_para:
                content.append(('para', '\n'.join(current_para)))
                current_para = []
            content.append(('h4', line[5:].strip()))
        # Tables
        elif '|' in line and not line.strip().startswith('|---'):
            if current_para:
                content.append(('para', '\n'.join(current_para)))
                current_para = []
            # Simple table detection - collect table rows
            if 'table_rows' not in locals():
                table_rows = []
            table_rows.append([cell.strip() for cell in line.split('|') if cell.strip()])
        elif line.strip() == '---' and 'table_rows' in locals() and table_rows:
            content.append(('table', table_rows))
            table_rows = []
        # Empty line
        elif not line.strip():
            if current_para:
                content.append(('para', '\n'.join(current_para)))
                current_para = []
        else:
            current_para.append(line)
    
    if current_para:
        content.append(('para', '\n'.join(current_para)))
    if 'table_rows' in locals() and table_rows:
        content.append(('table', table_rows))
    
    return content

def markdown_to_reportlab(text):
    """Convert markdown formatting to reportlab formatting."""
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)
    # Links (simplified)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    return text

def create_pdf():
    """Create PDF report."""
    # Read markdown
    with open(report_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Parse markdown
    parsed = parse_markdown(md_content)
    
    # Create PDF
    doc = SimpleDocTemplate(output_pdf, pagesize=A4,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=48,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=40,
        alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=28,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    course_style = ParagraphStyle(
        'CourseStyle',
        parent=styles['Normal'],
        fontSize=24,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=80,
        alignment=TA_CENTER
    )
    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontSize=20,
        textColor=colors.black,
        spaceAfter=15,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    story = []
    
    # Title page
    story.append(Spacer(1, 3*inch))
    story.append(Paragraph("HANS", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Hardness-Adaptive Negative Sampling", subtitle_style))
    story.append(Paragraph("for Dense Passage Retrieval", subtitle_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("IR Project DS501", course_style))
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph("Utkarsh Kumar - 12241930", author_style))
    story.append(Paragraph("Rishabh Sahu - 12241500", author_style))
    story.append(PageBreak())
    
    # Content
    for item_type, content in parsed:
        if item_type == 'h1':
            story.append(Spacer(1, 0.3*inch))
            p = Paragraph(markdown_to_reportlab(content), styles['Heading1'])
            story.append(p)
            story.append(Spacer(1, 0.2*inch))
        elif item_type == 'h2':
            story.append(Spacer(1, 0.25*inch))
            p = Paragraph(markdown_to_reportlab(content), styles['Heading2'])
            story.append(p)
            story.append(Spacer(1, 0.15*inch))
        elif item_type == 'h3':
            story.append(Spacer(1, 0.2*inch))
            p = Paragraph(markdown_to_reportlab(content), styles['Heading3'])
            story.append(p)
            story.append(Spacer(1, 0.1*inch))
        elif item_type == 'h4':
            story.append(Spacer(1, 0.15*inch))
            p = Paragraph(markdown_to_reportlab(content), styles['Heading4'])
            story.append(p)
            story.append(Spacer(1, 0.1*inch))
        elif item_type == 'para':
            if content.strip():
                text = markdown_to_reportlab(content.strip())
                p = Paragraph(text, styles['Normal'])
                story.append(p)
                story.append(Spacer(1, 0.12*inch))
        elif item_type == 'code':
            story.append(Spacer(1, 0.1*inch))
            p = Preformatted(content, styles['Code'])
            story.append(p)
            story.append(Spacer(1, 0.15*inch))
        elif item_type == 'table':
            if content:
                # Create table
                data = content
                t = Table(data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]),
                ]))
                story.append(t)
                story.append(Spacer(1, 0.2*inch))
    
    # Build PDF
    print("Creating PDF...")
    doc.build(story)
    print(f"PDF created successfully: {output_pdf}")
    return output_pdf

if __name__ == "__main__":
    try:
        create_pdf()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

