"""
PDF Report Generator
Creates comprehensive evaluation reports in PDF format
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from typing import Dict, List, Any
from collections import defaultdict
import textwrap


class PDFReportGenerator:
    """Generate PDF evaluation reports"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        
        self.styles.add(ParagraphStyle(
            name='CodeBlock',
            parent=self.styles['Normal'],
            fontName='Courier',
            fontSize=8,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=6,
            spaceAfter=6,
            backColor=colors.HexColor('#f5f5f5')
        ))
    
    def generate_report(self, results: List[Dict], framework_info: Dict[str, Any], output_path: str):
        """Generate comprehensive PDF report"""
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )
        
        story = []
        
        # Title Page
        story.extend(self._generate_title_page(framework_info))
        
        # Executive Summary
        story.extend(self._generate_executive_summary(results))
        
        story.append(PageBreak())
        
        # Detailed Results
        story.extend(self._generate_detailed_results(results))
        
        # Build PDF
        doc.build(story)
        
        print(f"PDF report generated: {output_path}")
    
    def _generate_title_page(self, framework_info: Dict) -> List:
        """Generate title page"""
        
        elements = []
        
        elements.append(Paragraph("LLM Evaluation Report", self.styles['Title']))
        elements.append(Spacer(1, 20))
        
        info_text = f"""<b>Generated:</b> {framework_info.get('timestamp', 'Unknown')}<br/>
<b>Test Models:</b> {', '.join(framework_info.get('test_models', ['Unknown']))}<br/>
<b>Judge Model:</b> {framework_info.get('judge_model', 'Unknown')}<br/>
<b>DeepEval:</b> {'Enabled' if framework_info.get('deepeval_enabled') else 'Disabled'}<br/>"""
        
        if framework_info.get('deepeval_enabled'):
            metrics = ', '.join(framework_info.get('deepeval_metrics', []))
            info_text += f"<b>DeepEval Metrics:</b> {metrics}<br/>"
        
        elements.append(Paragraph(info_text, self.styles['Normal']))
        elements.append(Spacer(1, 30))
        
        return elements
    
    def _generate_executive_summary(self, results: List[Dict]) -> List:
        """Generate executive summary"""
        
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['Heading2']))
        elements.append(Spacer(1, 12))
        
        # Group by model
        by_model = defaultdict(list)
        for result in results:
            model_key = f"{result['provider']}/{result['model_id']}"
            by_model[model_key].append(result)
        
        # Summary table
        summary_data = [['Model', 'Tests', 'Pass Rate', 'Avg Score', 'Avg Latency']]
        
        for model_key, model_results in by_model.items():
            total = len(model_results)
            passed = sum(1 for r in model_results if r['status'] == 'PASS')
            avg_score = sum(r['overall_score'] for r in model_results) / total
            avg_latency = sum(r['latency_ms'] for r in model_results) / total
            
            # Truncate long model names
            display_model = model_key if len(model_key) <= 40 else model_key[:37] + "..."
            
            summary_data.append([
                display_model,
                str(total),
                f"{(passed/total*100):.1f}%",
                f"{avg_score:.2f}",
                f"{avg_latency:.0f}ms"
            ])
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 0.7*inch, 0.9*inch, 0.9*inch, 1*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
        
        # Add category-wise performance table
        elements.append(Paragraph("Performance by Category", self.styles['Heading3']))
        elements.append(Spacer(1, 8))
        
        # Calculate category statistics
        category_stats = self._calculate_category_statistics(results)
        
        category_data = [['Category', 'Tests', 'Pass Rate', 'Avg Score', 'Status']]
        
        for category, stats in sorted(category_stats.items()):
            status = 'PASS' if stats['avg_score'] >= 0.7 else 'WARNING' if stats['avg_score'] >= 0.5 else 'FAIL'
            
            category_data.append([
                category.upper(),
                str(stats['total']),
                f"{stats['pass_rate']:.1f}%",
                f"{stats['avg_score']:.2f}",
                status
            ])
        
        category_table = Table(category_data, colWidths=[2*inch, 0.8*inch, 1*inch, 1*inch, 1*inch])
        category_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        
        elements.append(category_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _calculate_category_statistics(self, results: List[Dict]) -> Dict[str, Dict]:
        """Calculate statistics by category"""
        
        by_category = defaultdict(list)
        for result in results:
            by_category[result['category']].append(result)
        
        category_stats = {}
        for category, cat_results in by_category.items():
            total = len(cat_results)
            passed = sum(1 for r in cat_results if r['status'] == 'PASS')
            avg_score = sum(r['overall_score'] for r in cat_results) / total if total > 0 else 0
            pass_rate = (passed / total * 100) if total > 0 else 0
            
            category_stats[category] = {
                'total': total,
                'passed': passed,
                'pass_rate': pass_rate,
                'avg_score': avg_score
            }
        
        return category_stats
    
    def _generate_detailed_results(self, results: List[Dict]) -> List:
        """Generate detailed test results"""
        
        elements = []
        
        # Group by model and category
        by_model = defaultdict(lambda: defaultdict(list))
        for result in results:
            model_key = f"{result['provider']}/{result['model_id']}"
            by_model[model_key][result['category']].append(result)
        
        for model_key, categories in by_model.items():
            elements.append(Paragraph(f"Model: {model_key}", self.styles['Heading2']))
            elements.append(Spacer(1, 12))
            
            for category, cat_results in categories.items():
                elements.append(Paragraph(f"Category: {category.upper()}", self.styles['Heading3']))
                elements.append(Spacer(1, 8))
                
                for idx, result in enumerate(cat_results, 1):
                    elements.extend(self._generate_single_test(result, idx))
                
                elements.append(Spacer(1, 15))
            
            elements.append(PageBreak())
        
        return elements
    
    def _generate_single_test(self, result: Dict, test_num: int) -> List:
        """Generate single test display"""
        
        elements = []
        
        # Test header
        status_color = self._get_status_color(result['status'])
        header = f"<b>Test {test_num}: {result['test_name']}</b> - <font color='{status_color}'>{result['status']}</font> | Score: {result['overall_score']:.2f}"
        elements.append(Paragraph(header, self.styles['Heading4']))
        elements.append(Spacer(1, 6))
        
        # Input/Output
        elements.append(Paragraph("<b>Input Prompt:</b>", self.styles['Normal']))
        prompt_text = self._wrap_text(result['prompt'], 100)
        elements.append(Paragraph(prompt_text, self.styles['CodeBlock']))
        elements.append(Spacer(1, 6))
        
        elements.append(Paragraph("<b>AI Response:</b>", self.styles['Normal']))
        response_text = self._wrap_text(result['response'], 100)
        elements.append(Paragraph(response_text, self.styles['CodeBlock']))
        elements.append(Spacer(1, 8))
        
        # Method scores
        if result.get('method_scores'):
            score_data = [['Method', 'Score', 'Status']]
            
            for method, score in result['method_scores'].items():
                status = "PASS" if score >= 0.7 else "WARNING" if score >= 0.5 else "FAIL"
                score_data.append([
                    method.replace('_', ' ').title(),
                    f"{score:.2f}",
                    status
                ])
            
            score_table = Table(score_data, colWidths=[2*inch, 1*inch, 1*inch])
            score_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER')
            ]))
            
            elements.append(score_table)
            elements.append(Spacer(1, 8))
        
        # Observations and recommendations
        elements.append(Paragraph(f"<b>Observations:</b> {result.get('observations', 'N/A')}", self.styles['Normal']))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(f"<b>Recommendations:</b> {result.get('recommendations', 'N/A')}", self.styles['Normal']))
        
        # Performance metrics
        elements.append(Spacer(1, 6))
        perf_text = f"Tokens: {result.get('input_tokens', 0)}->{result.get('output_tokens', 0)} | Latency: {result.get('latency_ms', 0):.0f}ms | Time: {result.get('timestamp', 'N/A')}"
        elements.append(Paragraph(perf_text, self.styles['Normal']))
        
        elements.append(Spacer(1, 12))
        
        return elements
    
    def _wrap_text(self, text: str, width: int = 80) -> str:
        """Wrap text for better display"""
        if not text:
            return ""
        
        # Escape HTML and wrap
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        wrapped_lines = textwrap.wrap(text, width=width)
        return '<br/>'.join(wrapped_lines[:10])  # Limit to 10 lines
    
    def _get_status_color(self, status: str) -> str:
        """Get color for status"""
        colors_map = {
            'PASS': 'green',
            'FAIL': 'red',
            'WARNING': 'orange'
        }
        return colors_map.get(status, 'black')