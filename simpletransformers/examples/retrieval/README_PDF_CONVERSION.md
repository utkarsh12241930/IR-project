# PDF Conversion Instructions

The HTML report has been generated: `HANS_IR_Project_DS501.html`

## Method 1: Browser Print to PDF (Recommended)

1. Open `HANS_IR_Project_DS501.html` in your web browser (Chrome, Edge, Firefox)
2. Press `Ctrl+P` (or `Cmd+P` on Mac) to open the print dialog
3. Select **"Save as PDF"** or **"Microsoft Print to PDF"** as the destination
4. Make sure "Background graphics" is enabled in print settings
5. Click **"Save"**
6. Save as `HANS_IR_Project_DS501.pdf`

## Method 2: Using Pandoc (if installed)

```bash
pandoc HANS_IR_Project_DS501.html -o HANS_IR_Project_DS501.pdf
```

## Method 3: Online HTML to PDF Converter

1. Go to an online converter like:
   - https://www.ilovepdf.com/html-to-pdf
   - https://www.freeconvert.com/html-to-pdf
2. Upload `HANS_IR_Project_DS501.html`
3. Download the PDF

## Method 4: Using Python (if dependencies available)

If you have `pdfkit` and `wkhtmltopdf` installed:
```bash
pip install pdfkit
# Download wkhtmltopdf from https://wkhtmltopdf.org/downloads.html
pdfkit.from_file('HANS_IR_Project_DS501.html', 'HANS_IR_Project_DS501.pdf')
```

**Note**: The HTML file is already formatted with proper styling and includes the title page with:
- Heading: HANS
- Course: IR Project DS501
- Authors: Utkarsh Kumar - 12241930 and Rishabh Sahu - 12241500

