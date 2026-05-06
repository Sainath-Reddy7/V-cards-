# 📇 V-cards- (Business Card AI Extractor)

An advanced AI pipeline designed to convert business card images into structured, validated JSON data using Gemini 2.5 Flash and Qwen2.5-VL.

## 🚀 Key Features
* **Intelligent Extraction**: Captures Name, Organisation, Designation, and Contact details with high precision.
* **Regional Support**: Built-in mapping for major cities and zip codes in **Andhra Pradesh** and **Telangana**.
* **Dual-Engine Logic**: Optimized to use Google Gemini API for cloud reasoning and local OCR fallbacks[cite: 4, 5].
* **Structured Output**: Saves data in clean JSON format for easy integration into CRM or Excel[cite: 9].

## 📂 Dataset Status
* **Location**: `C:\Users\bsain\Desktop\Camera`
* **Count**: 54 Business Card Images.

## 🛠️ Setup
1. Clone this repo.
2. Install requirements: `pip install google-genai pydantic transformers`.
3. Run the extractor: `python model2.py --api-key "YOUR_KEY"`.