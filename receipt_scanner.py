import io
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReceiptScanner:
    @staticmethod
    def preprocess_image(image_data):
        """Enhanced image preprocessing for better OCR accuracy"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance image quality
            image = ReceiptScanner.enhance_image_quality(image)
            
            # Convert PIL to OpenCV format
            open_cv_image = np.array(image)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            
            # Apply multiple preprocessing techniques
            processed_image = ReceiptScanner.apply_image_processing(open_cv_image)
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            raise e
    
    @staticmethod
    def enhance_image_quality(pil_image):
        """Enhance PIL image quality before OpenCV processing"""
        # Resize if image is too small
        width, height = pil_image.size
        if width < 800 or height < 600:
            scale_factor = max(800/width, 600/height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(2.0)
        
        # Apply slight denoising
        pil_image = pil_image.filter(ImageFilter.MedianFilter())
        
        return pil_image
    
    @staticmethod
    def apply_image_processing(cv_image):
        """Apply OpenCV image processing techniques"""
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding for better text extraction
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Alternative: OTSU thresholding
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        adaptive_thresh = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        
        # Try both thresholding methods and return the one with better text detection
        return ReceiptScanner.select_best_processed_image(adaptive_thresh, otsu_thresh)
    
    @staticmethod
    def select_best_processed_image(adaptive_thresh, otsu_thresh):
        """Select the better processed image based on text detection confidence"""
        try:
            # Quick OCR test on both images to see which gives better results
            adaptive_text = pytesseract.image_to_string(adaptive_thresh, config='--psm 6')
            otsu_text = pytesseract.image_to_string(otsu_thresh, config='--psm 6')
            
            # Simple heuristic: choose the one with more alphanumeric characters
            adaptive_score = len(re.findall(r'[a-zA-Z0-9]', adaptive_text))
            otsu_score = len(re.findall(r'[a-zA-Z0-9]', otsu_text))
            
            return adaptive_thresh if adaptive_score >= otsu_score else otsu_thresh
            
        except:
            # If OCR fails, default to adaptive threshold
            return adaptive_thresh
    
    @staticmethod
    def extract_text(image_data):
        """Extract text using OCR with multiple configurations"""
        try:
            processed_image = ReceiptScanner.preprocess_image(image_data)
            
            # Try multiple OCR configurations
            ocr_configs = [
                '--psm 6',  # Uniform block of text
                '--psm 4',  # Single column of text
                '--psm 1',  # Automatic page segmentation with OSD
                '--psm 3',  # Fully automatic page segmentation
            ]
            
            best_text = ""
            best_confidence = 0
            
            for config in ocr_configs:
                try:
                    # Get text with confidence scores
                    data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Calculate average confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    # Get the actual text
                    text = pytesseract.image_to_string(processed_image, config=config)
                    
                    # Choose the result with highest confidence
                    if avg_confidence > best_confidence and len(text.strip()) > 10:
                        best_confidence = avg_confidence
                        best_text = text
                        
                except Exception as config_error:
                    logger.warning(f"OCR config {config} failed: {config_error}")
                    continue
            
            # If no good result, try with basic config
            if not best_text.strip():
                best_text = pytesseract.image_to_string(processed_image)
            
            logger.info(f"OCR completed with confidence: {best_confidence:.2f}")
            return best_text
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ReceiptScanner.extract_text_fallback(image_data)
    
    @staticmethod
    def extract_text_fallback(image_data):
        """
        Fallback method when Tesseract is not available.
        Provides helpful error message and guidance instead of fake data.
        """
        try:
            # Try to at least load and analyze the image
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            
            # Return informative message about the uploaded image
            return f"""OCR_ERROR: Tesseract not installed

IMAGE_INFO:
- Format: {image.format}
- Dimensions: {width}x{height} pixels

"""

        except Exception as img_error:
            return f"""ERROR: Cannot process image
            
DETAILS: {str(img_error)}

POSSIBLE_CAUSES:
- Corrupted image file
- Unsupported image format
- Memory issues

SOLUTION: Try uploading a different image (JPG, PNG, JPEG)"""
        
    @staticmethod
    def parse_receipt(text):
        """Enhanced receipt parsing with improved pattern recognition"""
        # Check if this is an error message from fallback
        if text.startswith("OCR_ERROR:") or text.startswith("ERROR:"):
            return {
                "total_amount": None,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M"),
                "vendor": "OCR Error - Tesseract not installed",
                "raw_text": text
            }
        
        # Clean the text
        clean_text = ReceiptScanner.clean_text(text)
        
        # Extract total amount
        total_amount = ReceiptScanner.extract_total_amount(clean_text)
        
        # Extract date and time
        date_info = ReceiptScanner.extract_date_time(clean_text)
        
        # Extract vendor
        vendor = ReceiptScanner.extract_vendor(clean_text)
        
        return {
            "total_amount": total_amount,
            "date": date_info["date"],
            "time": date_info["time"],
            "vendor": vendor,
            "raw_text": text
        }
    
    @staticmethod
    def clean_text(text):
        """Clean and normalize the OCR text"""
        # Remove extra whitespace and normalize
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        clean_text = '\n'.join(lines)
        
        # Fix common OCR errors
        replacements = {
            '|': 'I',  # Common OCR mistake
            '0': 'O',  # In vendor names
            '5': 'S',  # In vendor names (context-dependent)
        }
        
        # Apply replacements carefully (only for non-numeric contexts)
        return clean_text
    
    @staticmethod
    def extract_total_amount(text):
        """Extract total amount with comprehensive pattern matching"""
        # Enhanced patterns for various receipt formats
        total_patterns = [
            # Standard total patterns
            r'total[\s:]*[₹\$]?\s*(\d+[,.]?\d*\.?\d{0,2})',
            r'grand\s*total[\s:]*[₹\$]?\s*(\d+[,.]?\d*\.?\d{0,2})',
            r'net\s*total[\s:]*[₹\$]?\s*(\d+[,.]?\d*\.?\d{0,2})',
            r'final\s*total[\s:]*[₹\$]?\s*(\d+[,.]?\d*\.?\d{0,2})',
            
            # Amount patterns
            r'amount[\s:]*[₹\$]?\s*(\d+[,.]?\d*\.?\d{0,2})',
            r'amount\s*due[\s:]*[₹\$]?\s*(\d+[,.]?\d*\.?\d{0,2})',
            r'total\s*amount[\s:]*[₹\$]?\s*(\d+[,.]?\d*\.?\d{0,2})',
            
            # Balance patterns
            r'balance[\s:]*[₹\$]?\s*(\d+[,.]?\d*\.?\d{0,2})',
            r'balance\s*due[\s:]*[₹\$]?\s*(\d+[,.]?\d*\.?\d{0,2})',
            
            # Payment patterns
            r'cash[\s:]*[₹\$]?\s*(\d+[,.]?\d*\.?\d{0,2})',
            r'paid[\s:]*[₹\$]?\s*(\d+[,.]?\d*\.?\d{0,2})',
            
            # Currency symbol patterns
            r'[₹\$]\s*(\d+[,.]?\d*\.?\d{0,2})\s*$',  # Currency at start, amount at end of line
            r'(\d+[,.]?\d*\.?\d{0,2})\s*[₹\$]?\s*$',  # Amount at end of line
            
            # Standalone number patterns (last resort)
            r'^(\d+\.\d{2})$',  # Exact decimal format on its own line
        ]

        total_amount = None
        found_amounts = []
        
        # Split text into lines for line-by-line analysis
        lines = text.split('\n')
        
        for pattern in total_patterns:
            for line in lines:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    try:
                        # Clean the match (remove commas, handle decimal points)
                        clean_amount = re.sub(r'[,\s]', '', match)
                        amount = float(clean_amount)
                        found_amounts.append((amount, pattern, line))
                    except ValueError:
                        continue
        
        # Select the most likely total amount
        if found_amounts:
            # Sort by amount (descending) and pattern priority
            found_amounts.sort(key=lambda x: (-x[0], total_patterns.index(x[1])))
            total_amount = found_amounts[0][0]
            logger.info(f"Found total amount: {total_amount} from pattern: {found_amounts[0][1]}")
        
        return total_amount
    
    @staticmethod
    def extract_date_time(text):
        """Extract date and time information"""
        # Enhanced date patterns supporting multiple formats
        date_patterns = [
            # dd/mm/yyyy and dd-mm-yyyy formats
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            # yyyy-mm-dd format
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            # dd/mm/yy format
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2})',
            # Month name formats
            r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})',
        ]
        
        # Time patterns
        time_patterns = [
            r'(\d{1,2}:\d{2}:\d{2})',  # HH:MM:SS
            r'(\d{1,2}:\d{2})\s*(?:am|pm)?',  # HH:MM with optional AM/PM
            r'time[\s:]*(\d{1,2}:\d{2})',  # Time: HH:MM
        ]
        
        date = datetime.now().strftime("%Y-%m-%d")
        time = datetime.now().strftime("%H:%M")
        
        # Extract date
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # Normalize date format
                try:
                    # Try different parsing formats
                    for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%y', '%d-%m-%y']:
                        try:
                            parsed_date = datetime.strptime(date_str, fmt)
                            date = parsed_date.strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            continue
                except:
                    date = date_str  # Keep original if parsing fails
                break
        
        # Extract time
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                time = match.group(1)
                break
        
        return {"date": date, "time": time}
    
    @staticmethod
    def extract_vendor(text):
        """Extract vendor/organization name with improved accuracy"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return "Unknown Vendor"
        
        # Enhanced vendor detection patterns
        vendor_indicators = [
            r'restaurant', r'cafe', r'coffee', r'hotel', r'store', r'shop', r'mart',
            r'retail', r'mall', r'plaza', r'center', r'centre', r'ltd', r'limited',
            r'inc', r'corp', r'company', r'enterprises', r'services', r'digital',
            r'pharmacy', r'medical', r'clinic', r'hospital', r'foods', r'kitchen'
        ]
        
        vendor = "Unknown Vendor"
        
        # Strategy 1: Look for first meaningful line (not starting with numbers/symbols)
        for line in lines[:5]:  # Check first 5 lines
            line_clean = re.sub(r'[^\w\s]', '', line).strip()
            if (len(line_clean) > 3 and 
                not re.match(r'^\d', line_clean) and  # Doesn't start with number
                not re.match(r'^(date|time|receipt|bill|invoice)', line_clean.lower()) and
                len(line_clean.split()) <= 6):  # Not too many words
                vendor = line.strip()
                break
        
        # Strategy 2: Look for lines containing vendor indicators
        if vendor == "Unknown Vendor":
            for line in lines[:10]:
                line_lower = line.lower()
                for indicator in vendor_indicators:
                    if indicator in line_lower:
                        vendor = line.strip()
                        break
                if vendor != "Unknown Vendor":
                    break
        
        # Strategy 3: Use the first non-empty line as fallback
        if vendor == "Unknown Vendor" and lines:
            first_line = lines[0].strip()
            if len(first_line) > 2 and not first_line.isdigit():
                vendor = first_line
        
        # Clean up the vendor name
        vendor = re.sub(r'[^\w\s&\'-]', '', vendor).strip()
        vendor = ' '.join(vendor.split())  # Normalize whitespace
        
        return vendor if vendor else "Unknown Vendor"
    
    @staticmethod
    def scan_receipt(image_data):
        text = ReceiptScanner.extract_text(image_data)
        receipt_data = ReceiptScanner.parse_receipt(text)
        return receipt_data