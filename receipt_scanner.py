import io
import cv2
import numpy as np
import pytesseract
from PIL import Image
from datetime import datetime
import re

class ReceiptScanner:
    @staticmethod
    def preprocess_image(image_data):
        image = Image.open(io.BytesIO(image_data))
        open_cv_image = np.array(image)

        if len(open_cv_image.shape) == 3:
            open_cv_image = open_cv_image[:, :, ::-1].copy()

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        denoised = cv2.medianBlur(thresh, 5)
        
        return denoised
    
    @staticmethod
    def extract_text(image):
        try:
            processed_image = ReceiptScanner.preprocess_image(image)
            text = pytesseract.image_to_string(processed_image)
            return text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""
        
    @staticmethod
    def parse_receipt(text):
        total_pattertotal_patterns = [
            r'total[\s:]*[\$]?(\d+\.\d{2})',
            r'amount[\s:]*[\$]?(\d+\.\d{2})',
            r'balance[\s:]*[\$]?(\d+\.\d{2})',
            r'[\$](\d+\.\d{2})[^0-9]*$'
        ]

        total_amount = None
        for pattern in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                total_amount = float(match.group(1))
                break

        date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(\d{4}[/-]\d{1,2}[/-]\d{1,2})'
        date_match = re.search(date_pattern, text)
        date = date_match.group(0) if date_match else datetime.now().strftime("%Y-%m-%d")

        vendor_patterns = [
            r'at\s+([^\n]+)\n',
            r'from\s+([^\n]+)\n',
            r'^([^\n]+)\n',
        ]

        vendor = "Unknown Vendor"
        for pattern in vendor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                vendor = match.group(1).strip()
                break
        
        return {
            "total_amount": total_amount,
            "date": date,
            "vendor": vendor,
            "raw_text": text
        }
    
    @staticmethod
    def scan_receipt(image_data):
        text = ReceiptScanner.extract_text(image_data)
        receipt_data = ReceiptScanner.parse_receipt(text)
        return receipt_data