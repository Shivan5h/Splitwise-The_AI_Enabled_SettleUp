from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
import json

# Import modules
from models import *
from receipt_scanner import ReceiptScanner
from expense_categorizer import ExpenseCategorizer
from settlement_optimizer import SettlementOptimizer
from smart_reminder import SmartReminder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Enhanced Expense Sharing API",
    description="A Smart Expense Management System with AI Features",
    version="1.0.0"
)

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock database (in production, use a real database)
users_db = {}
groups_db = {}
expenses_db = {}
debts_db = {}
reminders_db = {}

# API Endpoints
@app.get("/")
async def root():
    return {"message": "AI-Enhanced Expense Sharing System API"}

@app.post("/users/", response_model=dict)
async def create_user(user: UserCreate):
    """Create a new user"""
    user_id = str(len(users_db) + 1)
    users_db[user_id] = {
        "id": user_id,
        **user.dict(),
        "created_at": datetime.now().isoformat()
    }
    return {"user_id": user_id, **users_db[user_id]}

@app.get("/users/{user_id}", response_model=dict)
async def get_user(user_id: str):
    """Get user details"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[user_id]

@app.post("/groups/", response_model=dict)
async def create_group(group: GroupCreate):
    """Create a new group"""
    group_id = str(len(groups_db) + 1)
    groups_db[group_id] = {
        "id": group_id,
        "name": group.name,
        "members": group.member_ids,
        "created_at": datetime.now().isoformat()
    }
    return {"group_id": group_id, **groups_db[group_id]}

@app.get("/groups/{group_id}", response_model=dict)
async def get_group(group_id: str):
    """Get group details"""
    if group_id not in groups_db:
        raise HTTPException(status_code=404, detail="Group not found")
    return groups_db[group_id]

@app.post("/expenses/", response_model=dict)
async def create_expense(expense: ExpenseCreate):
    """Create a new expense"""
    expense_id = str(len(expenses_db) + 1)
    
    # Auto-categorize if not provided
    category = expense.category
    if not category:
        category = ExpenseCategorizer.categorize(expense.description)
    
    expenses_db[expense_id] = {
        "id": expense_id,
        **expense.dict(),
        "category": category,
        "created_at": datetime.now().isoformat()
    }
    
    # Calculate debts from this expense
    amount_per_person = expense.amount / len(expense.split_between)
    for debtor in expense.split_between:
        if debtor != expense.paid_by:
            debt_id = f"{expense_id}_{debtor}"
            debts_db[debt_id] = {
                "debtor": debtor,
                "creditor": expense.paid_by,
                "amount": amount_per_person,
                "expense_id": expense_id,
                "created_at": datetime.now().isoformat(),
                "status": "pending"
            }
    
    return {"expense_id": expense_id, **expenses_db[expense_id]}

@app.post("/scan-receipt/", response_model=dict)
async def scan_receipt(file: UploadFile = File(...)):
    """Scan a receipt and extract information"""
    try:
        # Read image file
        image_data = await file.read()
        
        # Extract text from image
        receipt_data = ReceiptScanner.scan_receipt(image_data)
        
        # Categorize based on vendor
        category = ExpenseCategorizer.categorize(receipt_data["vendor"])
        
        return {
            "success": True,
            "data": {
                **receipt_data,
                "category": category
            }
        }
    except Exception as e:
        logger.error(f"Receipt scanning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Receipt scanning failed: {str(e)}")

@app.get("/groups/{group_id}/settlements", response_model=dict)
async def calculate_settlements(group_id: str):
    """Calculate optimal settlements for a group"""
    if group_id not in groups_db:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Get all expenses for this group
    group_expenses = [
        expense for expense in expenses_db.values() 
        if set(expense["split_between"]).intersection(groups_db[group_id]["members"])
    ]
    
    # Calculate optimal settlements
    settlement_data = SettlementOptimizer.optimize_settlements(group_expenses)
    
    return {
        "group_id": group_id,
        **settlement_data
    }

@app.get("/reminders/generate", response_model=List[Reminder])
async def generate_reminders(threshold_days: int = 7, threshold_amount: float = 10.0):
    """Generate reminders for overdue debts"""
    reminders = SmartReminder.generate_reminders(debts_db, threshold_days, threshold_amount)
    
    # Store reminders
    for reminder in reminders:
        reminder_id = f"rem_{len(reminders_db) + 1}"
        reminders_db[reminder_id] = {
            "id": reminder_id,
            **reminder,
            "created_at": datetime.now().isoformat()
        }
    
    return list(reminders_db.values())

@app.get("/expenses/categorize", response_model=dict)
async def categorize_expense(description: str):
    """Categorize an expense based on description"""
    category = ExpenseCategorizer.categorize(description)
    return {"description": description, "category": category}

@app.get("/debts/", response_model=Dict[str, Any])
async def list_debts(user_id: Optional[str] = None):
    """List all debts, optionally filtered by user"""
    if user_id:
        user_debts = {
            debt_id: debt for debt_id, debt in debts_db.items()
            if debt["debtor"] == user_id or debt["creditor"] == user_id
        }
        return user_debts
    return debts_db

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)