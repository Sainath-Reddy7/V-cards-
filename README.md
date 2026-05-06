# 📇 V-cards- (Business Card AI Extractor)

An advanced AI pipeline designed to convert images of business cards into structured, validated JSON data. This system leverages the **Gemini 2.5 Flash** API for high-level reasoning and local OCR fallbacks for robustness.

## 🚀 Key Features

*   **Hybrid OCR Engine**: Combines local OCR (EasyOCR or PyTesseract) with LLM-based vision capabilities to ensure maximum data recovery.
*   **Structured Data Enforcement**: Uses **Pydantic** models to strictly enforce JSON schemas, ensuring that extracted data is consistent and ready for database entry.
*   **Intelligent Validation**: Includes field-level validation for emails (regex matching) and phone numbers (length and character checking).
*   **Regional Awareness**: Pre-integrated with a mapping of major cities and zip codes for **Andhra Pradesh** and **Telangana** to verify and clean location data.
*   **Batch Processing**: Engineered to handle the full dataset of 54 images in the `Camera/` directory with automated retry logic for API rate limits.

## 🛠️ Technical Architecture

The system operates in a three-stage pipeline:
1.  **Image Pre-processing**: Uses **Pillow** to prepare images for local OCR or direct API transmission.
2.  **Extraction**:
    *   **Primary**: Direct Vision-to-JSON using `gemini-2.5-flash`.
    *   **Secondary/Fallback**: Local text extraction via `easyocr` followed by LLM-based text parsing.
3.  **Post-Processing**: Data cleaning to fix common OCR artifacts (e.g., confusing 'O' with '0' in emails) and appending regional metadata.

## 📂 Project Structure

*   `model2.py`: The core execution script containing the AI logic and Pydantic schemas.
*   `ocr_check.py`: Utility script to verify local environment readiness for OCR libraries.
*   `ap_telangana_city_zips.json`: Regional dataset used to validate zip codes and city names.
*   `visiting_cards_extracted.json`: Final structured output from the 54-card dataset.
*   `Camera/`: Source directory containing the raw business card images.

## 🚦 Getting Started

### Prerequisites
*   Python 3.12+
*   NVIDIA GPU (Optional, for faster local OCR)
*   Google Gemini API Key

### Installation
```bash
pip install google-genai pydantic pillow easyocr pytesseract


### Usage
To process the entire dataset, run:
python model2.py --api-key "YOUR_GEMINI_API_KEY" --dataset-path "./Camera" --delay 10



📊 Extraction Performance


The current pipeline successfully extracts up to 15 distinct fields, including:

Identity: First Name, Last Name, Designation

Company: Organisation, GSTIN, CIN, or registration IDs

Contact: Mobile (WhatsApp), Landline, Email, Website

Geography: Full Address, City, State, Zip Code



