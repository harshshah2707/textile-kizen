# utils/pdf_generator.py
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.graphics.shapes import Drawing, Rect, Circle, Line, String as DString

def generate_roll_pdf(roll_data, output_path):
    """
    Generates a professional PDF inspection report for a fabric roll using reportlab.
    roll_data contains:
      - roll: dict representing database row in `rolls`
      - defects: list of dicts representing database rows in `defects`
    """
    roll = roll_data["roll"]
    defects = roll_data["defects"]
    
    # Establish document template
    doc = SimpleDocTemplate(
        output_path, 
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    meta_label_style = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        textColor=colors.HexColor('#475569')
    )
    
    meta_val_style = ParagraphStyle(
        'MetaValue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=12,
        textColor=colors.HexColor('#0F172A')
    )
    
    tbl_hdr_style = ParagraphStyle(
        'TableHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white
    )
    
    tbl_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#1E293B')
    )

    tbl_cell_bold_style = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#1E293B')
    )

    story = []
    
    # --- PAGE HEADER ---
    story.append(Paragraph("KIZEN ENGINEERING", title_style))
    story.append(Paragraph(
        "Fabric Quality Inspection Certificate  •  Innovation Is Our Tradition  •  <font color='#2563EB'><u>https://kizen.co.in</u></font>", 
        ParagraphStyle('SubTitle', fontName='Helvetica-Bold', fontSize=9, textColor=colors.HexColor('#475569'), spaceAfter=15)
    ))
    
    # --- METADATA ROW ---
    # Fetch material-specific width (default 1800mm if not logged)
    meta_data = [
        [
            Paragraph("Roll Number:", meta_label_style), Paragraph(str(roll["roll_number"]), meta_val_style),
            Paragraph("Material Type:", meta_label_style), Paragraph(str(roll["material_name"]), meta_val_style)
        ],
        [
            Paragraph("Operator:", meta_label_style), Paragraph(str(roll["operator_name"]), meta_val_style),
            Paragraph("Inspection Date:", meta_label_style), Paragraph(str(roll["started_at"]), meta_val_style)
        ],
        [
            Paragraph("Scanned Length:", meta_label_style), Paragraph(f"{roll['length_meters']:.2f} meters", meta_val_style),
            Paragraph("Quality Grade:", meta_label_style), Paragraph(
                f"<font color='{'green' if roll['grade'] == 'FIRST QUALITY' else 'red'}'><b>{roll['grade']}</b></font>", 
                meta_val_style
            )
        ],
        [
            Paragraph("Total Points:", meta_label_style), Paragraph(f"{roll['total_points']} pts", meta_val_style),
            Paragraph("Points / 100m²:", meta_label_style), Paragraph(f"{roll['points_per_100m']:.2f} pts", meta_val_style)
        ]
    ]
    
    meta_table = Table(meta_data, colWidths=[100, 160, 100, 160])
    meta_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LINEBELOW', (0,-1), (-1,-1), 1, colors.HexColor('#E2E8F0')),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))
    
    # --- 2D DEFECT CUTTING MAP ---
    story.append(Paragraph("2D Defect Cutting Map (ASTM D5430 Reference)", section_style))
    story.append(Paragraph(
        "The diagram below maps all detected defects along the width (X-axis, mm) and length (Y-axis, meters) of the fabric roll. "
        "Use this schematic to plan optimal cuts and minimize material waste.", 
        ParagraphStyle('MapDesc', parent=styles['Normal'], fontSize=8.5, textColor=colors.HexColor('#64748B'), spaceAfter=8)
    ))
    
    # Map layout parameters
    map_w = 520
    map_h = 100
    dwg = Drawing(map_w, map_h + 20)
    
    # Background representing the roll
    dwg.add(Rect(0, 15, map_w, map_h, fillColor=colors.HexColor('#F8FAFC'), strokeColor=colors.HexColor('#CBD5E1'), strokeWidth=1))
    
    # Draw vertical grid lines (every 20% of width)
    for i in range(1, 5):
        x = (map_w / 5) * i
        dwg.add(Line(x, 15, x, 15 + map_h, strokeColor=colors.HexColor('#E2E8F0'), strokeWidth=0.5))
    
    # Draw horizontal center line
    dwg.add(Line(0, 15 + (map_h/2), map_w, 15 + (map_h/2), strokeColor=colors.HexColor('#E2E8F0'), strokeWidth=0.5))
    
    # Ruler labels (Length: Left = 0m, Right = Total Roll Length)
    dwg.add(DString(5, 2, "Start (0.0 m)", fontSize=8, fontName="Helvetica-Bold", fillColor=colors.HexColor('#475569')))
    dwg.add(DString(map_w - 90, 2, f"End ({roll['length_meters']:.1f} m)", fontSize=8, fontName="Helvetica-Bold", fillColor=colors.HexColor('#475569')))
    dwg.add(DString(map_w / 2 - 35, 2, "Fabric Flow →", fontSize=8, fontName="Helvetica", fillColor=colors.HexColor('#94A3B8')))
    
    # Width indicators (0 to width_mm on sides)
    dwg.add(DString(-2, 18, "0", fontSize=7, fontName="Helvetica", fillColor=colors.HexColor('#64748B')))
    dwg.add(DString(-2, 15 + map_h - 4, "W", fontSize=7, fontName="Helvetica-Bold", fillColor=colors.HexColor('#64748B')))
    
    # Mark defects as red circles on the map
    total_len = max(0.1, roll['length_meters'])
    # Assume 1800mm width if fabric width not found
    width_mm = 1800
    if len(defects) > 0:
        # Try finding width from first defect metadata or database
        pass
        
    for d in defects:
        # Map Y (distance_meters) to X coordinate of the drawing (0 to map_w)
        dist_pct = min(1.0, max(0.0, d["distance_meters"] / total_len))
        cx = dist_pct * map_w
        
        # Map X (bbox_x / location coordinate) to Y coordinate of drawing (15 to 15 + map_h)
        # Parse x coord from string coordinate if bbox not stored, or use raw bbox_x
        bx = d.get("bbox_x", 0)
        # If camera slice is 4096px wide, normalize pixel X to drawing Y
        # In case we don't have pixel to mm, let's map pixel coordinates (0-4096) to height
        y_pct = min(1.0, max(0.0, bx / 4096.0))
        cy = 15 + (y_pct * map_h)
        
        # Color coding by severity/type
        dot_color = colors.HexColor('#EF4444') # Red for holes/critical
        if d["defect_type"] in ["stain", "Needle mark", "Pinched fabric"]:
            dot_color = colors.HexColor('#EA580C') # Orange
            
        # Draw dot
        dwg.add(Circle(cx, cy, 3.5, fillColor=dot_color, strokeColor=colors.white, strokeWidth=0.5))
        
    story.append(dwg)
    story.append(Spacer(1, 15))
    
    # --- DEFECT LOG TABLE ---
    story.append(Paragraph("Inspection Defect Inventory", section_style))
    
    if len(defects) == 0:
        story.append(Paragraph("No defects detected on this fabric roll.", ParagraphStyle('NoDef', parent=styles['Normal'], textColor=colors.HexColor('#10B981'), fontName='Helvetica-Bold', fontSize=10)))
    else:
        # Table headers
        table_data = [[
            Paragraph("Defect ID", tbl_hdr_style),
            Paragraph("Meter Mark", tbl_hdr_style),
            Paragraph("Defect Type", tbl_hdr_style),
            Paragraph("Confidence", tbl_hdr_style),
            Paragraph("Defect Size (mm)", tbl_hdr_style),
            Paragraph("ASTM Points", tbl_hdr_style)
        ]]
        
        # Calculate ASTM points helper locally
        def calculate_pts(dtype, size):
            if dtype == 'hole':
                return 2 if size <= 25 else 4
            else:
                if size <= 75: return 1
                if size <= 150: return 2
                if size <= 230: return 3
                return 4
                
        for idx, d in enumerate(defects):
            pts = calculate_pts(d["defect_type"], d["size_mm"])
            table_data.append([
                Paragraph(f"D-{d['id']}", tbl_cell_bold_style),
                Paragraph(f"{d['distance_meters']:.2f} m", tbl_cell_style),
                Paragraph(str(d["defect_type"]), tbl_cell_style),
                Paragraph(f"{d['confidence'] * 100:.0f}%", tbl_cell_style),
                Paragraph(f"{d['size_mm']:.1f} mm", tbl_cell_style),
                Paragraph(f"<b>{pts} pts</b>", tbl_cell_style)
            ])
            
        tbl = Table(table_data, colWidths=[70, 90, 110, 80, 110, 60])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ]))
        
        story.append(tbl)
        
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "<b>Kizen Engineering</b> — Innovation Is Our Tradition  •  <font color='#2563EB'>https://kizen.co.in</font>  •  Industrial Textile Inspection Suite",
        ParagraphStyle('FooterNotice', fontName='Helvetica', fontSize=8, textColor=colors.HexColor('#64748B'), alignment=1)
    ))
    
    # Build Document
    doc.build(story)
    print(f"[PDF] Quality report generated successfully at: {output_path}")
