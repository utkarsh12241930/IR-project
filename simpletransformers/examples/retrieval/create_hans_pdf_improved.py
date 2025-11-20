"""
Create PDF from LaTeX content - Improved version
Properly formatted academic report with better LaTeX parsing
"""
import os
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

script_dir = os.path.dirname(os.path.abspath(__file__))
tex_file = os.path.join(script_dir, "hans_report.tex")
output_pdf = os.path.join(script_dir, "HANS_Report_DS501.pdf")

def clean_latex_text(text):
    """Remove LaTeX commands and convert to plain text."""
    if not text:
        return ""
    
    # Handle escaped characters first
    text = text.replace('\\%', '%')
    text = text.replace('\\&', '&')
    text = text.replace('\\#', '#')
    text = text.replace('\\$', '$')
    
    # Remove LaTeX commands with braces
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    # Remove standalone commands
    text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
    # Remove remaining braces
    text = re.sub(r'\{([^}]*)\}', r'\1', text)
    # Remove math mode
    text = re.sub(r'\$([^$]*)\$', r'\1', text)
    # Replace citations
    text = re.sub(r'\\cite\{[^}]+\}', '[CITATION]', text)
    # Remove line breaks
    text = re.sub(r'\\\\', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_latex_content(tex_content):
    """Parse LaTeX content into structured sections."""
    content = {
        'title': '',
        'authors': [],
        'abstract': '',
        'keywords': '',
        'sections': []
    }
    
    # Extract title
    title_match = re.search(r'\\title\{([^}]+)\}', tex_content)
    if title_match:
        content['title'] = clean_latex_text(title_match.group(1))
    
    # Extract authors
    author_matches = re.findall(r'\\author\{([^}]+)\}', tex_content)
    content['authors'] = [clean_latex_text(a) for a in author_matches]
    
    # Extract abstract
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex_content, re.DOTALL)
    if abstract_match:
        content['abstract'] = clean_latex_text(abstract_match.group(1))
    
    # Extract keywords
    keywords_match = re.search(r'\\keywords\{([^}]+)\}', tex_content)
    if keywords_match:
        content['keywords'] = clean_latex_text(keywords_match.group(1))
    
    # Extract main content
    main_match = re.search(r'\\maketitle(.*?)\\begin\{thebibliography\}', tex_content, re.DOTALL)
    if not main_match:
        return content
    
    main_text = main_match.group(1)
    
    # Find all sections, subsections, and subsubsections
    section_pattern = r'\\(section|subsection|subsubsection)\{([^}]+)\}'
    matches = list(re.finditer(section_pattern, main_text))
    
    for i, match in enumerate(matches):
        level = 1 if match.group(1) == 'section' else (2 if match.group(1) == 'subsection' else 3)
        title = clean_latex_text(match.group(2))
        
        # Get content until next section
        start_pos = match.end()
        end_pos = matches[i+1].start() if i+1 < len(matches) else len(main_text)
        section_content = main_text[start_pos:end_pos].strip()
        
        # Clean and split into paragraphs
        paragraphs = []
        for para in section_content.split('\n\n'):
            para = para.strip()
            if para and not para.startswith('%') and not para.startswith('\\') and len(para) > 20:
                cleaned = clean_latex_text(para)
                if cleaned:
                    paragraphs.append(cleaned)
        
        if title or paragraphs:
            content['sections'].append({
                'level': level,
                'title': title,
                'paragraphs': paragraphs
            })
    
    return content

def create_pdf():
    """Create PDF from LaTeX content."""
    print("Reading LaTeX file...")
    with open(tex_file, 'r', encoding='utf-8') as f:
        tex_content = f.read()
    
    print("Parsing LaTeX content...")
    content = parse_latex_content(tex_content)
    
    # Create PDF
    doc = SimpleDocTemplate(output_pdf, pagesize=A4,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    # Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.black,
        spaceAfter=8,
        alignment=TA_CENTER
    )
    
    abstract_heading_style = ParagraphStyle(
        'AbstractHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.black,
        spaceAfter=10,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    abstract_style = ParagraphStyle(
        'AbstractStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.black,
        spaceAfter=20,
        alignment=TA_JUSTIFY,
        fontName='Helvetica',
        leftIndent=20,
        rightIndent=20
    )
    
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=12,
        spaceBefore=20,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    subsection_style = ParagraphStyle(
        'SubsectionStyle',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#333333'),
        spaceAfter=10,
        spaceBefore=15,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    subsubsection_style = ParagraphStyle(
        'SubsubsectionStyle',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=colors.HexColor('#555555'),
        spaceAfter=8,
        spaceBefore=12,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.black,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        fontName='Helvetica',
        leading=14
    )
    
    story = []
    
    # Title page
    story.append(Spacer(1, 1.5*inch))
    if content['title']:
        story.append(Paragraph(content['title'], title_style))
        story.append(Spacer(1, 0.4*inch))
    
    # Authors
    for author in content['authors']:
        story.append(Paragraph(author, author_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Abstract
    if content['abstract']:
        story.append(Paragraph("Abstract", abstract_heading_style))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(content['abstract'], abstract_style))
        story.append(Spacer(1, 0.2*inch))
    
    # Keywords
    if content['keywords']:
        story.append(Paragraph(f"<b>Keywords:</b> {content['keywords']}", normal_style))
        story.append(PageBreak())
    
    # Process sections
    print(f"Processing {len(content['sections'])} sections...")
    for section in content['sections']:
        # Add section heading
        if section['level'] == 1:
            story.append(Spacer(1, 0.2*inch))
            if section['title']:
                story.append(Paragraph(section['title'], section_style))
                story.append(Spacer(1, 0.15*inch))
        elif section['level'] == 2:
            story.append(Spacer(1, 0.15*inch))
            if section['title']:
                story.append(Paragraph(section['title'], subsection_style))
                story.append(Spacer(1, 0.1*inch))
        else:
            if section['title']:
                story.append(Paragraph(section['title'], subsubsection_style))
                story.append(Spacer(1, 0.08*inch))
        
        # Add paragraphs
        for para in section['paragraphs']:
            if para:
                story.append(Paragraph(para, normal_style))
                story.append(Spacer(1, 0.1*inch))
    
    # Build PDF
    print("Building PDF...")
    doc.build(story)
    print(f"\nPDF created successfully!")
    print(f"  Location: {output_pdf}")
    return output_pdf

if __name__ == "__main__":
    try:
        create_pdf()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

