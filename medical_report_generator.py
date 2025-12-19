import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO

class MedicalReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
        
    def _create_custom_styles(self):
        """Create custom styles for the medical report"""
        custom_styles = {}
        
        # Title style
        custom_styles['ReportTitle'] = ParagraphStyle(
            'ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            textColor=colors.darkblue,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        # Header style
        custom_styles['SectionHeader'] = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.darkred,
            spaceAfter=12,
            spaceBefore=20
        )
        
        # Sub-header style
        custom_styles['SubHeader'] = ParagraphStyle(
            'SubHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.darkblue,
            spaceAfter=8,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        )
        
        # Patient info style
        custom_styles['PatientInfo'] = ParagraphStyle(
            'PatientInfo',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6
        )
        
        # Result style
        custom_styles['Result'] = ParagraphStyle(
            'Result',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.darkgreen,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        )
        
        # Warning style
        custom_styles['Warning'] = ParagraphStyle(
            'Warning',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.red,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        )
        
        return custom_styles
    
    def generate_prediction_report(self, patient_data, prediction_result, output_path, ai_recommendations=None):
        """Generate PDF report for breast cancer risk prediction with AI-generated recommendations"""
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Header
        story.append(Paragraph("🏥 BREAST CANCER RISK ASSESSMENT REPORT", self.custom_styles['ReportTitle']))
        story.append(Spacer(1, 20))
        
        # Report Info
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", self.custom_styles['PatientInfo']))
        story.append(Paragraph(f"<b>Report Type:</b> Breast Cancer Risk Prediction", self.custom_styles['PatientInfo']))
        story.append(Spacer(1, 20))
        
        # Patient Information Section
        story.append(Paragraph("👤 PATIENT INFORMATION", self.custom_styles['SectionHeader']))
        
        patient_info = [
            ['Parameter', 'Value'],
            ['Age', f"{patient_data.get('Age', 'N/A')} years"],
            ['BMI', f"{patient_data.get('BMI', 'N/A')} kg/m²"],
            ['Family History', 'Yes' if patient_data.get('Family_History') == 1 else 'No'],
            ['Genetic Mutation', 'Yes' if patient_data.get('Genetic_Mutation') == 1 else 'No'],
            ['Previous Breast Biopsy', 'Yes' if patient_data.get('Previous_Breast_Biopsy') == 1 else 'No']
        ]
        
        patient_table = Table(patient_info, colWidths=[2.5*inch, 2.5*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))
        
        # Risk Assessment Result
        story.append(Paragraph("📊 RISK ASSESSMENT RESULT", self.custom_styles['SectionHeader']))
        
        is_high_risk = "High Risk" in prediction_result
        result_style = self.custom_styles['Warning'] if is_high_risk else self.custom_styles['Result']
        
        story.append(Paragraph(f"<b>Risk Level:</b> {prediction_result}", result_style))
        story.append(Spacer(1, 15))
        
        # Risk Factors Analysis
        story.append(Paragraph("🔍 RISK FACTORS ANALYSIS", self.custom_styles['SectionHeader']))
        
        risk_factors = []
        if patient_data.get('Age', 0) > 50:
            risk_factors.append("• Age above 50 years increases risk")
        if patient_data.get('BMI', 0) > 25:
            risk_factors.append("• BMI above 25 may contribute to increased risk")
        if patient_data.get('Family_History') == 1:
            risk_factors.append("• Family history is a significant risk factor")
        if patient_data.get('Genetic_Mutation') == 1:
            risk_factors.append("• Genetic mutations significantly increase risk")
        if patient_data.get('Previous_Breast_Biopsy') == 1:
            risk_factors.append("• Previous breast biopsy indicates prior concerns")
        
        if risk_factors:
            story.append(Paragraph("<b>Identified Risk Factors:</b>", self.styles['Normal']))
            for factor in risk_factors:
                story.append(Paragraph(factor, self.styles['Normal']))
        else:
            story.append(Paragraph("• No major risk factors identified", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("💡 AI-ENHANCED ANALYSIS & RECOMMENDATIONS", self.custom_styles['SectionHeader']))
        
        if ai_recommendations:
            # Use AI-generated recommendations
            
            # Analysis Section
            story.append(Paragraph("🔍 Analysis:", self.custom_styles['SubHeader']))
            if ai_recommendations.get('refined_analysis'):
                story.append(Paragraph(ai_recommendations['refined_analysis'], self.styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Immediate Actions
            if ai_recommendations.get('immediate_actions'):
                story.append(Paragraph("⚡ Immediate Actions:", self.custom_styles['SubHeader']))
                for action in ai_recommendations['immediate_actions']:
                    story.append(Paragraph(f"• {action}", self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Diet Recommendations
            if ai_recommendations.get('diet_recommendations'):
                diet = ai_recommendations['diet_recommendations']
                story.append(Paragraph("🥗 Diet Recommendations:", self.custom_styles['SubHeader']))
                
                if diet.get('introduction'):
                    story.append(Paragraph(diet['introduction'], self.styles['Normal']))
                    story.append(Spacer(1, 10))
                
                if diet.get('foods_to_include'):
                    story.append(Paragraph("<b>Foods to Include:</b>", self.styles['Normal']))
                    for food in diet['foods_to_include']:
                        story.append(Paragraph(f"• {food}", self.styles['Normal']))
                    story.append(Spacer(1, 10))
                
                if diet.get('foods_to_avoid'):
                    story.append(Paragraph("<b>Foods to Limit/Avoid:</b>", self.styles['Normal']))
                    for food in diet['foods_to_avoid']:
                        story.append(Paragraph(f"• {food}", self.styles['Normal']))
                    story.append(Spacer(1, 10))
                
                if diet.get('meal_planning_tips'):
                    story.append(Paragraph("<b>Meal Planning Tips:</b>", self.styles['Normal']))
                    for tip in diet['meal_planning_tips']:
                        story.append(Paragraph(f"• {tip}", self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Lifestyle Changes
            if ai_recommendations.get('lifestyle_changes'):
                lifestyle = ai_recommendations['lifestyle_changes']
                story.append(Paragraph("🏃‍♀️ Lifestyle Changes:", self.custom_styles['SubHeader']))
                
                if lifestyle.get('exercise_recommendations'):
                    story.append(Paragraph("<b>Exercise:</b>", self.styles['Normal']))
                    for exercise in lifestyle['exercise_recommendations']:
                        story.append(Paragraph(f"• {exercise}", self.styles['Normal']))
                    story.append(Spacer(1, 10))
                
                if lifestyle.get('stress_management'):
                    story.append(Paragraph("<b>Stress Management:</b>", self.styles['Normal']))
                    for stress in lifestyle['stress_management']:
                        story.append(Paragraph(f"• {stress}", self.styles['Normal']))
                    story.append(Spacer(1, 10))
                
                if lifestyle.get('sleep_hygiene'):
                    story.append(Paragraph("<b>Sleep Hygiene:</b>", self.styles['Normal']))
                    for sleep in lifestyle['sleep_hygiene']:
                        story.append(Paragraph(f"• {sleep}", self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Medical Care
            if ai_recommendations.get('medical_care'):
                medical = ai_recommendations['medical_care']
                story.append(Paragraph("🩺 Medical Care:", self.custom_styles['SubHeader']))
                
                if medical.get('screening_schedule'):
                    story.append(Paragraph(f"<b>Screening Schedule:</b> {medical['screening_schedule']}", self.styles['Normal']))
                    story.append(Spacer(1, 10))
                
                if medical.get('specialist_consultations'):
                    story.append(Paragraph("<b>Specialist Consultations:</b>", self.styles['Normal']))
                    for specialist in medical['specialist_consultations']:
                        story.append(Paragraph(f"• {specialist}", self.styles['Normal']))
                    story.append(Spacer(1, 10))
                
                if medical.get('questions_for_doctor'):
                    story.append(Paragraph("<b>Questions for Your Doctor:</b>", self.styles['Normal']))
                    for question in medical['questions_for_doctor']:
                        story.append(Paragraph(f"• {question}", self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Monitoring & Tracking
            if ai_recommendations.get('monitoring_and_tracking'):
                story.append(Paragraph("📊 Monitoring & Support:", self.custom_styles['SubHeader']))
                story.append(Paragraph("<b>What to Monitor:</b>", self.styles['Normal']))
                for monitor in ai_recommendations['monitoring_and_tracking']:
                    story.append(Paragraph(f"• {monitor}", self.styles['Normal']))
                story.append(Spacer(1, 10))
            
            # Support Resources
            if ai_recommendations.get('support_resources'):
                story.append(Paragraph("<b>Support Resources:</b>", self.styles['Normal']))
                for resource in ai_recommendations['support_resources']:
                    story.append(Paragraph(f"• {resource}", self.styles['Normal']))
        
        else:
            # Fallback to basic recommendations if AI fails
            if is_high_risk:
                recommendations = [
                    "• Consult with an oncologist or breast specialist immediately",
                    "• Consider genetic counseling if genetic mutations are present",
                    "• Increase screening frequency (discuss with your doctor)",
                    "• Maintain a healthy lifestyle with regular exercise",
                    "• Follow a balanced diet rich in fruits and vegetables",
                    "• Limit alcohol consumption",
                    "• Perform monthly breast self-examinations",
                    "• Stay up to date with mammograms as recommended by your doctor"
                ]
            else:
                recommendations = [
                    "• Continue with routine screening as recommended for your age group",
                    "• Maintain a healthy lifestyle with regular exercise",
                    "• Follow a balanced diet rich in fruits and vegetables",
                    "• Perform monthly breast self-examinations",
                    "• Stay informed about breast health",
                    "• Consult your healthcare provider for personalized advice"
                ]
            
            for rec in recommendations:
                story.append(Paragraph(rec, self.styles['Normal']))
        
        story.append(Spacer(1, 30))
        
        # Disclaimer
        story.append(Paragraph("⚠️ IMPORTANT DISCLAIMER", self.custom_styles['SectionHeader']))
        disclaimer_text = """
        This report is generated by an AI-based risk assessment tool and is for informational purposes only. 
        It should not be considered as a medical diagnosis or substitute for professional medical advice. 
        Please consult with qualified healthcare professionals for proper medical evaluation and personalized 
        treatment recommendations. The accuracy of this assessment depends on the accuracy of the input data provided.
        """
        story.append(Paragraph(disclaimer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        return True
    
    def generate_detection_report(self, test_data, detection_result, output_path, ai_analysis=None, image_path=None):
        """Generate PDF report for cancer detection analysis with AI-generated content"""
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Header
        story.append(Paragraph("🔬 BREAST CANCER DETECTION ANALYSIS REPORT", self.custom_styles['ReportTitle']))
        story.append(Spacer(1, 20))
        
        # Report Info
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", self.custom_styles['PatientInfo']))
        story.append(Paragraph(f"<b>Report Type:</b> Breast Cancer Detection Analysis", self.custom_styles['PatientInfo']))
        story.append(Spacer(1, 20))
        
        # Test Parameters Section
        story.append(Paragraph("📋 TEST PARAMETERS", self.custom_styles['SectionHeader']))
        
        if test_data:
            test_info = [
                ['Parameter', 'Value', 'Unit'],
                ['Radius Mean', f"{test_data.get('radius_mean', 'N/A')}", 'units'],
                ['Texture Mean', f"{test_data.get('texture_mean', 'N/A')}", 'units'],
                ['Perimeter Mean', f"{test_data.get('perimeter_mean', 'N/A')}", 'units'],
                ['Area Mean', f"{test_data.get('area_mean', 'N/A')}", 'units²'],
                ['Smoothness Mean', f"{test_data.get('smoothness_mean', 'N/A')}", 'units']
            ]
            
            test_table = Table(test_info, colWidths=[2*inch, 1.5*inch, 1*inch])
            test_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(test_table)
        
        story.append(Spacer(1, 20))
        
        # Detection Result
        story.append(Paragraph("🎯 DETECTION RESULT", self.custom_styles['SectionHeader']))
        
        is_malignant = "Malignant" in detection_result
        result_style = self.custom_styles['Warning'] if is_malignant else self.custom_styles['Result']
        
        story.append(Paragraph(f"<b>Analysis Result:</b> {detection_result}", result_style))
        story.append(Spacer(1, 15))
        
        # Image Analysis (if provided)
        if image_path and os.path.exists(image_path):
            story.append(Paragraph("🖼️ IMAGE ANALYSIS", self.custom_styles['SectionHeader']))
            try:
                # Add image to report
                img = Image(image_path, width=3*inch, height=3*inch)
                story.append(img)
                story.append(Paragraph("Analyzed medical image", self.styles['Normal']))
            except:
                story.append(Paragraph("Image could not be processed for report", self.styles['Normal']))
            story.append(Spacer(1, 20))
        
        # AI-Enhanced Analysis and Recommendations
        if ai_analysis:
            story.append(Paragraph("🤖 AI-ENHANCED ANALYSIS & RECOMMENDATIONS", self.custom_styles['SectionHeader']))
            
            # Initial Message (if present)
            if ai_analysis.get('initial_message'):
                story.append(Paragraph(ai_analysis['initial_message'], self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Interpretation
            if ai_analysis.get('interpretation'):
                story.append(Paragraph("🩺 Clinical Interpretation:", self.custom_styles['SubHeader']))
                story.append(Paragraph(ai_analysis['interpretation'], self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Immediate Next Steps
            if ai_analysis.get('immediate_next_steps'):
                story.append(Paragraph("⚡ Immediate Next Steps:", self.custom_styles['SubHeader']))
                for step in ai_analysis['immediate_next_steps']:
                    story.append(Paragraph(f"• {step}", self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Medical Follow-up
            if ai_analysis.get('medical_follow_up'):
                story.append(Paragraph("🏥 Medical Follow-up:", self.custom_styles['SubHeader']))
                for follow in ai_analysis['medical_follow_up']:
                    story.append(Paragraph(f"• {follow}", self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Support Resources
            if ai_analysis.get('support_resources'):
                story.append(Paragraph("📞 Support Resources:", self.custom_styles['SubHeader']))
                for resource in ai_analysis['support_resources']:
                    story.append(Paragraph(f"• {resource}", self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Preventive Recommendations
            if ai_analysis.get('preventive_recommendations'):
                story.append(Paragraph("🛡️ Preventive Care & Monitoring:", self.custom_styles['SubHeader']))
                prev = ai_analysis['preventive_recommendations']
                
                if prev.get('introduction'):
                    story.append(Paragraph(prev['introduction'], self.styles['Normal']))
                    story.append(Spacer(1, 10))
                
                if prev.get('lifestyle_tips'):
                    story.append(Paragraph("<b>Lifestyle Recommendations:</b>", self.styles['Normal']))
                    for tip in prev['lifestyle_tips']:
                        story.append(Paragraph(f"• {tip}", self.styles['Normal']))
                    story.append(Spacer(1, 10))
                
                if prev.get('monitoring_guidelines'):
                    story.append(Paragraph("<b>Monitoring Guidelines:</b>", self.styles['Normal']))
                    for guideline in prev['monitoring_guidelines']:
                        story.append(Paragraph(f"• {guideline}", self.styles['Normal']))
                    story.append(Spacer(1, 15))
            
            # Important Reminders
            if ai_analysis.get('important_reminders'):
                story.append(Paragraph("💡 Important Reminders:", self.custom_styles['SubHeader']))
                for reminder in ai_analysis['important_reminders']:
                    story.append(Paragraph(f"• {reminder}", self.styles['Normal']))
                story.append(Spacer(1, 15))
            
        else:
            # Fallback to basic content if AI analysis not available
            story.append(Paragraph("🩺 CLINICAL INTERPRETATION", self.custom_styles['SectionHeader']))
            
            if is_malignant:
                interpretation = [
                    "• The analysis indicates characteristics consistent with malignant tissue",
                    "• Immediate medical consultation is strongly recommended",
                    "• Further diagnostic tests may be required for confirmation",
                    "• Early detection and treatment are crucial for optimal outcomes"
                ]
            else:
                interpretation = [
                    "• The analysis indicates characteristics consistent with benign tissue",
                    "• Continue with routine screening as recommended",
                    "• Monitor for any changes and report to healthcare provider",
                    "• Maintain regular breast health check-ups"
                ]
            
            for interp in interpretation:
                story.append(Paragraph(interp, self.styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Next Steps
            story.append(Paragraph("📅 RECOMMENDED NEXT STEPS", self.custom_styles['SectionHeader']))
            
            if is_malignant:
                next_steps = [
                    "1. Schedule immediate consultation with an oncologist",
                    "2. Discuss biopsy or additional imaging if recommended",
                    "3. Consider genetic counseling if family history is present",
                    "4. Prepare list of questions for medical consultation",
                    "5. Bring this report to your healthcare appointment"
                ]
            else:
                next_steps = [
                    "1. Continue with routine screening schedule",
                    "2. Maintain healthy lifestyle habits",
                    "3. Perform monthly self-examinations",
                    "4. Report any changes to healthcare provider",
                    "5. Follow up as recommended by your doctor"
                ]
            
            for step in next_steps:
                story.append(Paragraph(step, self.styles['Normal']))
        
        story.append(Spacer(1, 30))
        
        # Disclaimer
        story.append(Paragraph("⚠️ IMPORTANT DISCLAIMER", self.custom_styles['SectionHeader']))
        disclaimer_text = """
        This report is generated by an AI-based detection analysis tool and is for informational purposes only. 
        It should not be considered as a medical diagnosis or substitute for professional medical advice, diagnosis, 
        or treatment. Please consult with qualified healthcare professionals for proper medical evaluation and 
        personalized treatment recommendations. The accuracy of this analysis depends on the quality and accuracy 
        of the input data and images provided.
        """
        story.append(Paragraph(disclaimer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        return True

# Utility function to install required packages if not available
def install_reportlab():
    """Install reportlab if not available"""
    try:
        import reportlab
        return True
    except ImportError:
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
            return True
        except:
            return False

# Test the installation
if __name__ == "__main__":
    if install_reportlab():
        print("✅ ReportLab is available")
        # Test report generation
        generator = MedicalReportGenerator()
        
        # Test prediction report
        test_patient_data = {
            'Age': 45,
            'BMI': 26.5,
            'Family_History': 1,
            'Genetic_Mutation': 0,
            'Previous_Breast_Biopsy': 0
        }
        
        try:
            generator.generate_prediction_report(
                test_patient_data, 
                "⚠ High Risk of Developing Cancer",
                "test_prediction_report.pdf"
            )
            print("✅ Test prediction report generated")
        except Exception as e:
            print(f"❌ Error generating prediction report: {e}")
        
        # Test detection report
        test_detection_data = {
            'radius_mean': 14.5,
            'texture_mean': 18.2,
            'perimeter_mean': 92.3,
            'area_mean': 654.1,
            'smoothness_mean': 0.098
        }
        
        try:
            generator.generate_detection_report(
                test_detection_data,
                "✅ Benign (Non-Cancerous)",
                "test_detection_report.pdf"
            )
            print("✅ Test detection report generated")
        except Exception as e:
            print(f"❌ Error generating detection report: {e}")
    else:
        print("❌ ReportLab installation failed")