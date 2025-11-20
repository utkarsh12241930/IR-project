"""
Generate HTML version of the HANS project report with title page
Can be converted to PDF using browser (Print to PDF) or online tools
"""
import os
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
report_file = os.path.join(script_dir, "PROJECT_REPORT.md")
output_html = os.path.join(script_dir, "HANS_IR_Project_DS501.html")

def markdown_to_html(md_text):
    """Simple markdown to HTML converter."""
    html = md_text
    
    # Headers
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    
    # Bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    
    # Italic
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    
    # Code blocks
    html = re.sub(r'```(\w+)?\n(.*?)```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)
    
    # Inline code
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # Links
    html = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', html)
    
    # Tables (simple)
    lines = html.split('\n')
    in_table = False
    table_html = []
    result = []
    
    for line in lines:
        if '|' in line and not line.strip().startswith('|---'):
            if not in_table:
                in_table = True
                table_html = ['<table>']
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if cells:
                tag = 'th' if '---' in str(lines[lines.index(line)+1] if lines.index(line)+1 < len(lines) else '') else 'td'
                row = '<tr>' + ''.join(f'<{tag}>{cell}</{tag}>' for cell in cells) + '</tr>'
                table_html.append(row)
        else:
            if in_table:
                table_html.append('</table>')
                result.append('\n'.join(table_html))
                table_html = []
                in_table = False
            if line.strip() and not line.strip().startswith('|---'):
                result.append(line)
    
    if in_table:
        table_html.append('</table>')
        result.append('\n'.join(table_html))
    
    html = '\n'.join(result)
    
    # Paragraphs
    paragraphs = html.split('\n\n')
    html_paras = []
    for para in paragraphs:
        para = para.strip()
        if para and not para.startswith('<'):
            html_paras.append(f'<p>{para}</p>')
        else:
            html_paras.append(para)
    html = '\n\n'.join(html_paras)
    
    # Lists
    html = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    
    # Horizontal rules
    html = html.replace('---', '<hr>')
    
    return html

def create_html_report():
    """Generate HTML report with title page."""
    
    # Read the markdown report
    with open(report_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown_to_html(markdown_content)
    
    # Create full HTML document
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>HANS - IR Project DS501</title>
    <style>
        @media print {{
            @page {{
                size: A4;
                margin: 2.5cm;
            }}
            .title-page {{
                page-break-after: always;
            }}
            .content {{
                page-break-before: always;
            }}
        }}
        
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .title-page {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 80vh;
            text-align: center;
            page-break-after: always;
        }}
        
        .main-title {{
            font-size: 48px;
            font-weight: bold;
            margin-bottom: 40px;
            color: #1a1a1a;
        }}
        
        .subtitle {{
            font-size: 28px;
            margin-bottom: 20px;
            color: #2c3e50;
        }}
        
        .course-info {{
            font-size: 24px;
            margin: 60px 0;
            color: #34495e;
        }}
        
        .authors {{
            font-size: 20px;
            margin-top: 80px;
        }}
        
        .author-name {{
            margin: 15px 0;
            font-weight: bold;
        }}
        
        h1 {{
            font-size: 24px;
            margin-top: 30px;
            margin-bottom: 15px;
            color: #1a1a1a;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }}
        
        h2 {{
            font-size: 20px;
            margin-top: 25px;
            margin-bottom: 12px;
            color: #2c3e50;
        }}
        
        h3 {{
            font-size: 18px;
            margin-top: 20px;
            margin-bottom: 10px;
            color: #34495e;
        }}
        
        p {{
            margin-bottom: 12px;
            text-align: justify;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            page-break-inside: avoid;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            page-break-inside: avoid;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        
        ul, ol {{
            margin-left: 30px;
            margin-bottom: 12px;
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #ddd;
            margin: 30px 0;
        }}
        
        strong {{
            font-weight: bold;
        }}
        
        em {{
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="title-page">
        <div class="main-title">HANS</div>
        <div class="subtitle">Hardness-Adaptive Negative Sampling</div>
        <div class="subtitle">for Dense Passage Retrieval</div>
        <div class="course-info">IR Project DS501</div>
        <div class="authors">
            <div class="author-name">Utkarsh Kumar - 12241930</div>
            <div class="author-name">Rishabh Sahu - 12241500</div>
        </div>
    </div>
    
    <div class="content">
        {html_content}
    </div>
</body>
</html>"""
    
    # Write HTML file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"HTML report generated successfully: {output_html}")
    print("\nTo convert to PDF:")
    print("1. Open the HTML file in a browser")
    print("2. Press Ctrl+P (or Cmd+P on Mac)")
    print("3. Select 'Save as PDF' as the destination")
    print("4. Click 'Save'")
    print(f"\nOr use: pandoc {output_html} -o HANS_IR_Project_DS501.pdf")
    
    return output_html

if __name__ == "__main__":
    create_html_report()

