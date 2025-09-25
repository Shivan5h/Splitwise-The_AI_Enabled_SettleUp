# 🧾 Splitwise Receipt Scanner Demo

A Streamlit web application that demonstrates AI-powered receipt scanning and expense splitting functionality.

## Features

### 📱 Receipt Scanner
- **Automatic OCR**: Scans `image.jpg` and extracts key information
- **Smart Parsing**: Identifies total amount, vendor name, and date
- **Visual Interface**: Shows both original receipt image and extracted data

### 💸 Expense Creation
- **One-Click Expense**: Scan receipt and create expense automatically
- **User Management**: Select who paid and who to split between
- **Group Support**: Assign expenses to specific groups
- **Category Classification**: Organize expenses by category

### 👥 User & Group Management
- **User Overview**: View all registered users
- **Group Overview**: Manage expense groups
- **Real-time Data**: Live connection to FastAPI backend

## Quick Start

### Option 1: Automated Setup
```bash
# Run the demo script (Windows)
./run_demo.bat
# OR
./run_demo.ps1
```

### Option 2: Manual Setup
```bash
# 1. Start FastAPI server
python main.py

# 2. In another terminal, start Streamlit
streamlit run app.py
```

## Application URLs
- **FastAPI Server**: http://localhost:8000
- **Streamlit App**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

## Usage Instructions

### 1. Prepare Receipt Image
- Place your receipt image as `image.jpg` in the project directory
- Alternatively, use the upload feature in the app

### 2. Create Users (First Time)
Before using the expense features, create users via the FastAPI endpoints:
```bash
# Example API calls
curl -X POST "http://localhost:8000/users/" \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "email": "john@example.com", "phone": "1234567890"}'
```

### 3. Use the Scanner
1. Go to **Receipt Scanner** tab
2. Click **Scan Receipt** to extract data
3. Review the extracted information

### 4. Create Expenses
1. Go to **Create Expense from Receipt** tab
2. Select who paid and who to split between
3. Choose group and category
4. Click **Scan Receipt & Create Expense**

## Sample Receipt
The demo includes a sample receipt from "JUET CAFE GUNA" (₹12.00) that you can use for testing.

## API Endpoints

### Receipt Scanning
- `POST /scan-receipt/` - Scan image.jpg and return extracted data
- `POST /scan-receipt-create-expense/` - Scan and create expense

### User Management
- `POST /users/` - Create new user
- `GET /users/` - List all users
- `GET /users/{user_id}` - Get user details

### Expense Management
- `POST /expenses/` - Create expense
- `GET /expenses/` - List expenses
- `GET /groups/{group_id}/settlements` - Calculate settlements

## Technology Stack
- **Backend**: FastAPI, Python
- **Frontend**: Streamlit
- **OCR**: Tesseract, OpenCV
- **Image Processing**: PIL, NumPy

## Dependencies
```
fastapi
uvicorn
streamlit
opencv-python
pytesseract
pillow
numpy
pandas
requests
```

## Troubleshooting

### API Connection Issues
- Ensure FastAPI server is running on port 8000
- Check if any other services are using the ports

### OCR Issues
- Ensure Tesseract is installed on your system
- Verify image quality (clear text, good lighting)

### Image Upload Issues
- Supported formats: JPG, JPEG, PNG
- Maximum file size depends on system memory

## Future Enhancements
- [ ] Multiple receipt upload
- [ ] Receipt history management
- [ ] Advanced expense analytics
- [ ] Mobile app integration
- [ ] Real-time collaboration features