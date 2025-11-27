import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance, ImageFilter
import requests
import re
import json
from typing import Dict, List, Optional, Union
import logging
import io
import cv2
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available. Install with: pip install easyocr")

class DonutReceiptScanner:
    def __init__(self, model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2"):
        """
        Initialize Donut Receipt Scanner
        
        Args:
            model_name: Hugging Face model name for Donut
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Donut model loaded successfully")
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Advanced preprocessing for better OCR accuracy
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Convert to grayscale for better text detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding to handle varying lighting
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Convert back to PIL Image
        image = Image.fromarray(gray)
        
        # Convert back to RGB (3 channels) for model compatibility
        image = image.convert('RGB')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # Increased from 1.5
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)  # Increased from 1.2
        
        # Apply unsharp mask for better edge detection
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        return image
    
    def load_image(self, image_source: str) -> Image.Image:
        """
        Load image from file path or URL
        """
        try:
            if image_source.startswith('http'):
                response = requests.get(image_source, stream=True, timeout=30)
                image = Image.open(response.raw)
            else:
                image = Image.open(image_source)
            
            return self.preprocess_image(image)
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    def extract_receipt_data(self, image_source: str, max_length: int = 768) -> Dict:
        """
        Extract receipt data from image
        
        Args:
            image_source: Path to image or URL
            max_length: Maximum sequence length for generation
            
        Returns:
            Dictionary containing extracted receipt information
        """
        try:
            # Load and preprocess image
            image = self.load_image(image_source)
            
            # Prepare inputs
            task_prompt = "<s_cord-v2>"
            decoder_input_ids = self.processor.tokenizer(
                task_prompt, 
                add_special_tokens=False, 
                return_tensors="pt"
            ).input_ids
            
            # Process image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            
            # Generate output
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values.to(self.device),
                    decoder_input_ids=decoder_input_ids.to(self.device),
                    max_length=max_length,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=5,  # Increased for better accuracy
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                    output_scores=True,
                    length_penalty=1.0,
                    repetition_penalty=1.2,
                )
            
            # Decode output
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = self.clean_sequence(sequence)
            
            # Convert to JSON
            result = self.processor.token2json(sequence)
            
            # Post-process results
            processed_result = self.post_process_results(result)
            
            logger.info("Successfully extracted receipt data")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error extracting receipt data: {e}")
            return {
                "store_name": "Error",
                "total_amount": "0.00",
                "date": "Error",
                "subtotal": "0.00",
                "tax": "0.00",
                "items": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def clean_sequence(self, sequence: str) -> str:
        """
        Clean the generated sequence
        """
        # Remove special tokens
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.processor.tokenizer.pad_token, "")
        
        # Remove task prompt
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        return sequence
    
    def post_process_results(self, raw_result: Dict) -> Dict:
        """
        Post-process and validate extracted results
        """
        # Default values
        processed = {
            "store_name": self.extract_store_name(raw_result),
            "total_amount": self.extract_total_amount(raw_result),
            "date": raw_result.get("date", "Unknown"),
            "subtotal": raw_result.get("subtotal", "0.00"),
            "tax": raw_result.get("tax", "0.00"),
            "items": self.extract_items(raw_result),
            "currency": "USD",
            "confidence": 0.8,  # Placeholder for confidence score
            "raw_result": raw_result
        }
        
        # Validate and clean amounts
        processed["total_amount"] = self.clean_amount(processed["total_amount"])
        processed["subtotal"] = self.clean_amount(processed["subtotal"])
        processed["tax"] = self.clean_amount(processed["tax"])
        
        return processed
    
    def extract_store_name(self, result: Dict) -> str:
        """
        Extract store name from various possible fields with better validation
        """
        store_fields = ["company", "store", "shop", "vendor", "merchant", "name", "seller"]
        
        for field in store_fields:
            if field in result and result[field]:
                store_name = str(result[field]).strip()
                # Filter out common false positives
                if (store_name and 
                    store_name != "N/A" and 
                    len(store_name) > 1 and 
                    not re.match(r'^\d+$', store_name) and  # Not just numbers
                    not re.match(r'^[\d/:-]+$', store_name)):  # Not date/time format
                    # Clean up common OCR errors
                    store_name = re.sub(r'[^a-zA-Z0-9\s&.-]', '', store_name)
                    store_name = ' '.join(store_name.split())  # Normalize whitespace
                    if len(store_name) > 2:
                        return store_name
        
        return "Unknown Store"
    
    def extract_total_amount(self, result: Dict) -> str:
        """
        Extract total amount from various possible fields
        """
        total_fields = ["total", "total_paid", "amount", "grand_total"]
        
        for field in total_fields:
            if field in result and result[field]:
                amount = str(result[field]).strip()
                if amount and amount != "N/A":
                    return amount
        
        return "0.00"
    
    def extract_items(self, result: Dict) -> List[Dict]:
        """
        Extract line items from receipt
        """
        items = []
        
        if "items" in result and isinstance(result["items"], list):
            for item in result["items"]:
                if isinstance(item, dict):
                    cleaned_item = {
                        "description": item.get("description", ""),
                        "quantity": item.get("quantity", "1"),
                        "price": self.clean_amount(item.get("price", "0.00")),
                        "amount": self.clean_amount(item.get("amount", "0.00"))
                    }
                    items.append(cleaned_item)
        
        return items
    
    def clean_amount(self, amount: str) -> str:
        """
        Clean and format amount strings with validation to prevent time concatenation
        """
        if not amount or amount == "N/A":
            return "0.00"
        
        # Convert to string and remove whitespace
        amount = str(amount).strip()
        
        # Remove currency symbols but keep digits, dots, and commas
        amount = re.sub(r'[^\d.,]', '', amount)
        
        # Remove commas (thousand separators)
        amount = amount.replace(',', '')
        
        # Handle multiple dots (keep only the last one as decimal point)
        if amount.count('.') > 1:
            parts = amount.split('.')
            amount = ''.join(parts[:-1]) + '.' + parts[-1]
        
        # Validate: if no digits, return 0.00
        if not re.search(r'\d', amount):
            return "0.00"
        
        # Check for suspiciously long numbers (likely concatenated with time/date)
        # Valid amounts typically don't exceed 8 digits before decimal
        if '.' in amount:
            integer_part = amount.split('.')[0]
            if len(integer_part) > 8:
                # Likely concatenated - try to extract reasonable amount
                # Look for patterns like XXX06:56 becoming XXX0656
                # Common pattern: time format digits (00-59:00-59) at the end
                match = re.search(r'^(\d+?)([0-5]\d[0-5]\d)$', integer_part)
                if match:
                    amount = match.group(1)  # Take the part before time digits
                    logger.warning(f"Detected time concatenation, extracted: {amount}")
                else:
                    # If no pattern match, take first reasonable digits
                    amount = integer_part[:6]  # Limit to 6 digits max
        else:
            # No decimal point
            if len(amount) > 8:
                # Check for time pattern at end
                match = re.search(r'^(\d+?)([0-5]\d[0-5]\d)$', amount)
                if match:
                    amount = match.group(1)
                    logger.warning(f"Detected time concatenation, extracted: {amount}")
                else:
                    amount = amount[:6]
        
        # Ensure proper decimal format
        try:
            amount_float = float(amount) if '.' in amount else float(amount)
            # Sanity check: amounts over 1,000,000 are suspicious
            if amount_float > 1000000:
                logger.warning(f"Suspiciously large amount: {amount_float}, capping at 999999.99")
                amount_float = 999999.99
            # Format to 2 decimal places
            amount = f"{amount_float:.2f}"
        except ValueError:
            logger.error(f"Could not parse amount: {amount}")
            return "0.00"
        
        return amount
    
    def batch_process(self, image_paths: List[str]) -> List[Dict]:
        """
        Process multiple receipts in batch
        """
        results = []
        for path in image_paths:
            try:
                result = self.extract_receipt_data(path)
                result["file_path"] = path
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results.append({
                    "file_path": path,
                    "error": str(e),
                    "store_name": "Error",
                    "total_amount": "0.00"
                })
        
        return results

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "processor_type": type(self.processor).__name__,
            "model_type": type(self.model).__name__,
            "vocab_size": self.processor.tokenizer.vocab_size
        }

    def extract_receipt_from_pil(self, image: Image.Image, max_length: int = 768) -> Dict:
        """
        Extract receipt data when caller already has a PIL Image (e.g. from bytes).
        """
        try:
            # Ensure image is preprocessed similar to load_image
            image = self.preprocess_image(image)

            task_prompt = "<s_cord-v2>"
            decoder_input_ids = self.processor.tokenizer(
                task_prompt,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids

            pixel_values = self.processor(image, return_tensors="pt").pixel_values

            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values.to(self.device),
                    decoder_input_ids=decoder_input_ids.to(self.device),
                    max_length=max_length,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=2,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = self.clean_sequence(sequence)
            result = self.processor.token2json(sequence)
            processed_result = self.post_process_results(result)
            return processed_result
        except Exception as e:
            logger.error(f"Error extracting receipt from PIL image: {e}")
            raise


class ReceiptScanner:
    """High-level scanner for FastAPI.
    
    Uses Donut AI model with EasyOCR fallback for accurate receipt scanning.
    Returns: dict with total_amount, vendor, raw_text, items
    """
    @staticmethod
    def scan_receipt(image_bytes: bytes) -> Dict:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            return {"total_amount": 0.0, "vendor": "Error", "raw_text": "", "error": str(e)}

        # Try Donut AI first
        donut_result = None
        try:
            donut = DonutReceiptScanner()
            donut_result = donut.extract_receipt_from_pil(img)
            
            total_str = donut_result.get("total_amount", "0")
            vendor = donut_result.get("store_name", "Unknown")
            
            # Parse total amount
            try:
                total_val = float(total_str)
            except:
                total_val = 0.0
            
            # If Donut failed to extract meaningful data, try EasyOCR
            if (total_val == 0.0 or vendor == "Unknown Store") and EASYOCR_AVAILABLE:
                logger.info("Donut extraction incomplete, trying EasyOCR...")
                ocr_result = ReceiptScanner._extract_with_easyocr(img)
                
                # Use EasyOCR data if better
                if ocr_result["total_amount"] > 0 and total_val == 0:
                    total_val = ocr_result["total_amount"]
                    logger.info(f"Using EasyOCR total: {total_val}")
                if ocr_result["vendor"] != "Unknown" and vendor == "Unknown Store":
                    vendor = ocr_result["vendor"]
                    logger.info(f"Using EasyOCR vendor: {vendor}")
            
            logger.info(f"Final result: ${total_val} from {vendor}")
            return {
                "total_amount": total_val,
                "vendor": vendor,
                "raw_text": json.dumps(donut_result.get("raw_result", {})),
                "items": donut_result.get("items", []),
                "date": donut_result.get("date", "Unknown"),
                "method": "donut+ocr" if EASYOCR_AVAILABLE else "donut"
            }
        except Exception as e:
            logger.error(f"Receipt scanning failed: {e}")
            # Try EasyOCR as complete fallback
            if EASYOCR_AVAILABLE:
                try:
                    logger.info("Donut failed, using EasyOCR fallback")
                    return ReceiptScanner._extract_with_easyocr(img)
                except:
                    pass
            return {
                "total_amount": 0.0,
                "vendor": "Unknown",
                "raw_text": "",
                "items": [],
                "error": str(e)
            }
    
    @staticmethod
    def _extract_with_easyocr(img: Image.Image) -> Dict:
        """
        Fallback extraction using EasyOCR for better accuracy
        """
        try:
            # Initialize EasyOCR reader (supports English and Hindi for Indian receipts)
            reader = easyocr.Reader(['en', 'hi'], gpu=torch.cuda.is_available())
            
            # Convert PIL to numpy array
            img_array = np.array(img)
            
            # Perform OCR
            results = reader.readtext(img_array)
            
            # Extract text
            all_text = [text for (bbox, text, prob) in results]
            raw_text = ' '.join(all_text)
            
            # Find total amount - look for common patterns
            total_amount = 0.0
            total_patterns = [
                r'total[:\s]+(?:rs\.?|₹)?\s*([\d,]+\.?\d*)',
                r'grand\s+total[:\s]+(?:rs\.?|₹)?\s*([\d,]+\.?\d*)',
                r'amount[:\s]+(?:rs\.?|₹)?\s*([\d,]+\.?\d*)',
                r'₹\s*([\d,]+\.?\d*)\s*$',  # Currency at start of line
                r'rs\.?\s*([\d,]+\.?\d*)\s*$',  # Rs. at start
            ]
            
            for pattern in total_patterns:
                match = re.search(pattern, raw_text.lower())
                if match:
                    try:
                        amount_str = match.group(1).replace(',', '')
                        total_amount = float(amount_str)
                        if total_amount > 0:
                            logger.info(f"Found total with pattern: {pattern} = {total_amount}")
                            break
                    except:
                        continue
            
            # Find vendor name - usually first few lines, all caps or title case
            vendor = "Unknown"
            for i, (bbox, text, prob) in enumerate(results[:5]):  # Check first 5 lines
                text = text.strip()
                # Vendor names are usually longer than 3 chars, not just numbers
                if (len(text) > 3 and 
                    not re.match(r'^[\d/:-]+$', text) and
                    prob > 0.5):  # Confidence threshold
                    vendor = text
                    logger.info(f"Found vendor: {vendor}")
                    break
            
            return {
                "total_amount": total_amount,
                "vendor": vendor,
                "raw_text": raw_text,
                "items": [],
                "date": "Unknown",
                "method": "easyocr"
            }
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return {
                "total_amount": 0.0,
                "vendor": "Unknown",
                "raw_text": "",
                "items": [],
                "error": str(e)
            }