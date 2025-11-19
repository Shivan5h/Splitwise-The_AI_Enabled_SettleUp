import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance
import requests
import re
import json
from typing import Dict, List, Optional, Union
import logging
import io
import pytesseract

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        Preprocess image for better OCR results
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
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
                    num_beams=2,  # Increased for better accuracy
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                    output_scores=True,
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
        Extract store name from various possible fields
        """
        store_fields = ["company", "store", "shop", "vendor", "merchant"]
        
        for field in store_fields:
            if field in result and result[field]:
                store_name = str(result[field]).strip()
                if store_name and store_name != "N/A":
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
        Clean and format amount strings
        """
        if not amount or amount == "N/A":
            return "0.00"
        
        # Remove currency symbols and extra spaces
        amount = re.sub(r'[^\d.]', '', str(amount))
        
        # Ensure proper decimal format
        if '.' in amount:
            parts = amount.split('.')
            if len(parts) == 2:
                integer_part = parts[0]
                decimal_part = parts[1][:2].ljust(2, '0')
                amount = f"{integer_part}.{decimal_part}"
        else:
            amount = f"{amount}.00"
        
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
    
    Attempts Tesseract OCR first (faster, lighter), then Donut AI as fallback.
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

        # Try Tesseract first (faster, no GPU needed)
        try:
            ocr_text = pytesseract.image_to_string(img)
            
            # Extract amounts using regex
            amount_patterns = [
                r"total[:\s]*\$?([\d,]+\.\d{2})",  # Total: $123.45
                r"amount[:\s]*\$?([\d,]+\.\d{2})",  # Amount: $123.45
                r"\$([\d,]+\.\d{2})",               # $123.45
                r"([\d,]+\.\d{2})"                  # 123.45
            ]
            
            amounts = []
            text_lower = ocr_text.lower()
            for pattern in amount_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    try:
                        clean_amt = float(match.replace(',', ''))
                        if clean_amt > 0:
                            amounts.append(clean_amt)
                    except:
                        pass
            
            total_amount = max(amounts) if amounts else 0.0
            
            # Extract vendor (first non-empty line)
            vendor = "Unknown"
            for line in ocr_text.splitlines():
                line = line.strip()
                if line and len(line) > 2:
                    vendor = line[:50]  # Limit length
                    break
            
            if total_amount > 0:
                logger.info(f"Tesseract OCR: ${total_amount} from {vendor}")
                return {
                    "total_amount": total_amount,
                    "vendor": vendor,
                    "raw_text": ocr_text,
                    "items": [],
                    "method": "tesseract"
                }
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}. Trying Donut AI...")

        # Fallback to Donut AI (slower but more accurate)
        try:
            donut = DonutReceiptScanner()
            result = donut.extract_receipt_from_pil(img)
            
            total = result.get("total_amount", "0")
            try:
                total_val = float(re.sub(r"[^0-9.]", "", str(total)) or 0)
            except:
                total_val = 0.0
            
            logger.info(f"Donut AI: ${total_val} from {result.get('store_name', 'Unknown')}")
            return {
                "total_amount": total_val,
                "vendor": result.get("store_name", "Unknown"),
                "raw_text": json.dumps(result.get("raw_result", {})),
                "items": result.get("items", []),
                "method": "donut"
            }
        except Exception as e:
            logger.error(f"Both OCR methods failed: {e}")
            return {
                "total_amount": 0.0,
                "vendor": "Unknown",
                "raw_text": "",
                "items": [],
                "error": str(e)
            }