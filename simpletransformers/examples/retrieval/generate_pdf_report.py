"""
Generate PDF version of the HANS project report
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
report_file = os.path.join(script_dir, "PROJECT_REPORT.md")
output_pdf = os.path.join(script_dir, "HANS_IR_Project_DS501.pdf")

def create_pdf_report():
    """Generate PDF from markdown report with title page."""
    
    # Read the markdown report
    with open(report_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Create title page HTML
    title_page = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {
                size: A4;
                margin: 2.5cm;
            }
            body {
                font-family: 'Times New Roman', serif;
                line-height: 1.6;
                color: #333;
            }
            .title-page {
                page-break-after: always;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
                text-align: center;
            }
            .main-title {
                font-size: 48px;
                font-weight: bold;
                margin-bottom: 40px;
                color: #1a1a1a;
            }
            .subtitle {
                font-size: 28px;
                margin-bottom: 60px;
                color: #2c3e50;
            }
            .course-info {
                font-size: 24px;
                margin-bottom: 80px;
                color: #34495e;
            }
            .authors {
                font-size: 20px;
                margin-top: 100px;
            }
            .author-name {
                margin: 15px 0;
                font-weight: bold;
            }
            .content {
                page-break-before: always;
            }
            h1 {
                font-size: 24px;
                margin-top: 30px;
                margin-bottom: 15px;
                color: #1a1a1a;
                border-bottom: 2px solid #3498db;
                padding-bottom: 5px;
            }
            h2 {
                font-size: 20px;
                margin-top: 25px;
                margin-bottom: 12px;
                color: #2c3e50;
            }
            h3 {
                font-size: 18px;
                margin-top: 20px;
                margin-bottom: 10px;
                color: #34495e;
            }
            p {
                margin-bottom: 12px;
                text-align: justify;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                page-break-inside: avoid;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 10px;
                text-align: left;
            }
            th {
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            code {
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }
            pre {
                background-color: #f4f4f4;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                page-break-inside: avoid;
            }
            ul, ol {
                margin-left: 30px;
                margin-bottom: 12px;
            }
            li {
                margin-bottom: 8px;
            }
            .page-break {
                page-break-after: always;
            }
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
    """
    
    # Convert markdown to HTML using markdown library
    try:
        from markdown import markdown
        html_content = markdown(markdown_content, extensions=['tables', 'fenced_code', 'codehilite'])
    except ImportError:
        # Fallback: simple markdown conversion
        import re
        html_content = markdown_content
        # Convert headers
        html_content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
        html_content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
        html_content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
        # Convert bold
        html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
        # Convert code blocks
        html_content = re.sub(r'```(.+?)```', r'<pre><code>\1</code></pre>', html_content, flags=re.DOTALL)
        # Convert inline code
        html_content = re.sub(r'`(.+?)`', r'<code>\1</code>', html_content)
        # Convert line breaks
        html_content = html_content.replace('\n', '<br>\n')
    
    # Combine title page and content
    full_html = title_page + html_content + "</div></body></html>"
    
    # Generate PDF using weasyprint
    try:
        from weasyprint import HTML, CSS
        print("Generating PDF report...")
        HTML(string=full_html).write_pdf(
            output_pdf,
            stylesheets=[CSS(string="""
                @page {
                    size: A4;
                    margin: 2.5cm;
                }
            """)]
        )
        print(f"PDF report generated successfully: {output_pdf}")
        return output_pdf
    except ImportError:
        print("Error: weasyprint not installed. Installing required packages...")
        print("Please run: pip install weasyprint markdown")
        print("\nAlternatively, you can use pandoc:")
        print(f"pandoc {report_file} -o {output_pdf} --pdf-engine=wkhtmltopdf")
        return None

if __name__ == "__main__":
    create_pdf_report()
