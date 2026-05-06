import argparse
import os
import json
import time
import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator
from google import genai
from google.genai import types

# Local OCR imports may not be installed by default.
# Install with: pip install pillow easyocr
# easyocr is used by default, and pytesseract is a fallback.
try:
    from PIL import Image
    import easyocr
    OCR_AVAILABLE = True
    OCR_LIBRARY = 'easyocr'
except Exception:
    try:
        from PIL import Image
        import pytesseract
        OCR_AVAILABLE = True
        OCR_LIBRARY = 'pytesseract'
    except Exception:
        OCR_AVAILABLE = False

# 1. DEFINE DATA STRUCTURE (Strict Enforcement with Validation)
class BusinessCardSchema(BaseModel):
    first_name: Optional[str] = Field(description="First name + middle name + titles (Dr, Mr)")
    last_name: Optional[str] = Field(description="Surname / Family name")
    organisation: Optional[str] = Field(description="Full company or shop name")
    designation: Optional[str] = Field(description="Job title / Role")
    email: Optional[str] = Field(description="Valid email address")
    mobile: Optional[str] = Field(description="Mobile/WhatsApp number with country code")
    landline: Optional[str] = Field(description="Office / Landline number")
    website: Optional[str] = Field(description="Full URL as printed")
    full_address: Optional[str] = Field(description="Combined street, building, area")
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: Optional[str] = "India"
    gst_or_id: Optional[str] = Field(description="GSTIN, CIN, or registration numbers")
    other_details: List[str] = Field(default_factory=list, description="Slogans, product lists, or social media handles")

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            return None
        return v

    @field_validator('mobile', 'landline')
    @classmethod
    def validate_phone(cls, v):
        if v:
            if not re.match(r'^[\d\s\+\-\(\)]+$', v):
                return None
        return v

    @field_validator('website')
    @classmethod
    def validate_website(cls, v):
        if v and not v.startswith(('http://', 'https://', 'www.')):
            return 'https://' + v
        return v

# 2. SETUP
DEFAULT_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_DATASET_PATH = r"C:\Users\bsain\Desktop\Camera"
DEFAULT_OUTPUT_FILE = r"C:\Users\bsain\Desktop\visiting_cards_extracted.json"

client = None
TEXT_MODEL_CANDIDATES = [
    "gemini-flash-latest",
    "models/gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.5-flash"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract business card fields from images using local OCR + Gemini text models.")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="Folder containing business card images")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="JSON output file path")
    parser.add_argument("--image", help="Single image file path to process")
    parser.add_argument("--delay", type=int, default=60, help="Seconds to wait between API requests")
    parser.add_argument("--single", action="store_true", help="Only process the first image in the dataset directory")
    return parser.parse_args()


def initialize_client(api_key: str):
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Provide --api-key or set the GOOGLE_API_KEY environment variable."
        )
    return genai.Client(api_key=api_key)


def validate_api_key(client_obj):
    try:
        client_obj.models.list()
    except Exception as e:
        raise RuntimeError(
            "API key invalid or expired. Provide a valid Gemini API key via --api-key or GOOGLE_API_KEY."
        ) from e

if not OCR_AVAILABLE:
    print("WARNING: Local OCR is not available. Install Pillow and pytesseract, and make sure Tesseract OCR is on PATH.")
    print("Run: pip install pillow pytesseract")
    print("Then install Tesseract OCR from https://github.com/tesseract-ocr/tesseract.")

# 3. LOCAL OCR + TEXT EXTRACTION ENGINE

def extract_text_from_image(img_path):
    if not OCR_AVAILABLE:
        raise RuntimeError(
            "Local OCR libraries are not installed. Install pillow and easyocr or pytesseract."
        )

    image = Image.open(img_path)
    if OCR_LIBRARY == 'easyocr':
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(img_path, detail=0)
        text = '\n'.join(result)
    else:
        text = pytesseract.image_to_string(image, lang='eng')
    return text.strip()


def build_prompt(card_text: str) -> str:
    return f"""
You are an expert business card data extraction specialist. The text below is the result of OCR from a business card image.
Parse it exactly and fill ALL the JSON schema fields accurately. Do not miss any information.

Business card OCR text:
{card_text}

Rules:
- Extract EVERY piece of information visible in the text.
- For names: If there's a person's name, split into first_name (including titles like Dr., Mr., Ms., and middle names) and last_name (surname). If only one name, put it in first_name.
- For organisation: The company or business name.
- For designation: Job title or role.
- For email: Any email address, fix obvious OCR errors (e.g., 'O' to '0').
- For mobile: Primary phone number, especially if labeled as mobile or WhatsApp.
- For landline: Secondary phone number, if different from mobile.
- For website: Any URL or website address.
- For full_address: Combine all address lines into one string.
- For city, state, zip_code: Parse from the address if present.
- For country: Default to "India" if not specified.
- For gst_or_id: Any GST, CIN, PAN, or registration numbers.
- For other_details: Any additional info like slogans, social media, products, etc.

Examples:
- Text: "Dr. John Smith CEO ABC Corp john@abc.com +91 9876543210 www.abc.com Mumbai, Maharashtra 400001 GST: 22AAAAA0000A1Z5"
  Output: {{"first_name": "Dr. John", "last_name": "Smith", "organisation": "ABC Corp", "designation": "CEO", "email": "john@abc.com", "mobile": "+91 9876543210", "landline": null, "website": "https://www.abc.com", "full_address": "Mumbai, Maharashtra 400001", "city": "Mumbai", "state": "Maharashtra", "zip_code": "400001", "country": "India", "gst_or_id": "22AAAAA0000A1Z5", "other_details": []}}

- Text: "Rajesh Kumar Sharma Proprietor Sharma Enterprises rajesh@sharma.com 9876543210 040-1234567 www.sharma.com Delhi 110001 GSTIN: 07AAAP1234A1Z1 Facebook: @sharmaent"
  Output: {{"first_name": "Rajesh Kumar", "last_name": "Sharma", "organisation": "Sharma Enterprises", "designation": "Proprietor", "email": "rajesh@sharma.com", "mobile": "9876543210", "landline": "040-1234567", "website": "https://www.sharma.com", "full_address": "Delhi 110001", "city": "Delhi", "zip_code": "110001", "country": "India", "gst_or_id": "07AAAP1234A1Z1", "other_details": ["Facebook: @sharmaent"]}}

Return only valid JSON matching the schema. Fill as many fields as possible from the text.
"""


def post_process_extraction(card_text: str, extracted: BusinessCardSchema) -> BusinessCardSchema:
    """Post-process to fill missing fields from raw OCR text."""
    data = extracted.model_dump()

    # Extract emails if missing
    if not data.get('email'):
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', card_text)
        if email_match:
            data['email'] = email_match.group(0)

    # Extract phone numbers
    phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phones = re.findall(phone_pattern, card_text)
    phones = [p for p in phones if len(re.sub(r'\D', '', p)) >= 10]  # Filter valid lengths

    if phones:
        if not data.get('mobile'):
            data['mobile'] = phones[0]
        if len(phones) > 1 and not data.get('landline'):
            data['landline'] = phones[1]
        if len(phones) > 2:
            data['other_details'].extend([f"Phone: {p}" for p in phones[2:]])

    # Extract website if missing
    if not data.get('website'):
        url_match = re.search(r'\b(?:https?://|www\.)\S+\b', card_text, re.IGNORECASE)
        if url_match:
            data['website'] = url_match.group(0)

    # Extract GST/ID if missing
    if not data.get('gst_or_id'):
        gst_match = re.search(r'\b(?:GST|GSTIN|CIN|PAN)[\s:]*([A-Z0-9]+)\b', card_text, re.IGNORECASE)
        if gst_match:
            data['gst_or_id'] = gst_match.group(1)

    # Validate and return
    return BusinessCardSchema(**data)


def advanced_extract(img_path, retries=2):
    if not OCR_AVAILABLE:
        print("ERROR: OCR not available. Install pytesseract and Tesseract OCR to use this script.")
        return None

    try:
        card_text = extract_text_from_image(img_path)
    except Exception as e:
        print(f"   ❌ OCR failed for {os.path.basename(img_path)}: {e}")
        return None

    if not card_text:
        print(f"   ⚠️  OCR found no text in {os.path.basename(img_path)}")
        return None

    prompt = build_prompt(card_text)

    for attempt in range(retries + 1):
        for model_id in TEXT_MODEL_CANDIDATES:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=[card_text, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=BusinessCardSchema,
                        temperature=0.0
                    )
                )
                parsed = response.parsed
                if parsed and any(getattr(parsed, field) for field in ['first_name', 'last_name', 'organisation', 'email', 'mobile']):
                    try:
                        validated = BusinessCardSchema(**parsed.model_dump())
                        # Post-process to fill missing fields
                        validated = post_process_extraction(card_text, validated)
                        print(f"   ✅ Success with {model_id} on attempt {attempt + 1}")
                        return validated
                    except Exception as val_e:
                        print(f"   ⚠️  Validation failed for {model_id} on attempt {attempt + 1}: {val_e}")
                        continue
                else:
                    print(f"   ⚠️  No valid data from {model_id} on attempt {attempt + 1}")
                    continue
            except Exception as e:
                message = str(e)
                if '404' in message or 'NOT_FOUND' in message or 'not supported' in message:
                    print(f"   ⚠️  Model {model_id} unavailable, trying next model.")
                    continue
                if 'RESOURCE_EXHAUSTED' in message or 'quota' in message.lower():
                    print(f"   ⚠️  Quota issue on model {model_id}: {message}")
                    continue
                print(f"   ❌ Error with model {model_id} on attempt {attempt + 1}: {message}")
                continue
        if attempt < retries:
            time.sleep(2)
    return None

def main():
    global client

    args = parse_args()
    api_key = args.api_key or DEFAULT_API_KEY
    dataset_path = args.dataset_path
    output_file = args.output_file

    client = initialize_client(api_key)
    validate_api_key(client)

    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image does not exist: {args.image}")
        files = [args.image]
        dataset_path = os.path.dirname(args.image) or dataset_path
    else:
        if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist or is not a directory: {dataset_path}")
        files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not files:
        print("No image files found to process.")
        return

    if args.single and not args.image:
        files = files[:1]
    
    # Limit to 5 cards
    files = files[:5]

    all_results = []

    for filename in files:
        img_path = os.path.join(dataset_path, filename) if not args.image else filename
        data = advanced_extract(img_path)
        if data:
            record = data.model_dump()
            record['filename'] = os.path.basename(img_path)
            all_results.append(record)
            print(f"   ✅ Extracted: {record.get('first_name', 'N/A')} {record.get('last_name', 'N/A')} @ {record.get('organisation', 'N/A')}")
        else:
            print(f"   ❌ Failed to extract from {filename}")
        if len(files) > 1:
            time.sleep(args.delay)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print(f"\n🚀 DONE! {len(all_results)} cards extracted to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()