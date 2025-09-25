from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import os
from receipt_scanner import ReceiptScanner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Expense Sharing API",
    description="A Simple Expense Management System for Groups",
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

# ===== DATA MODELS =====
class UserCreate(BaseModel):
    name: str = Field(..., description="Full name of the user")
    email: str = Field(..., description="Email address of the user")
    phone: Optional[str] = Field(None, description="Phone number of the user")

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    phone: Optional[str]
    created_at: str

class GroupCreate(BaseModel):
    name: str = Field(..., description="Name of the group")
    member_ids: List[str] = Field(..., description="List of user IDs in the group")
    type: Optional[str] = Field("general", description="Type of group: roommates, travel, friends, family, work, sports")

class GroupResponse(BaseModel):
    id: str
    name: str
    members: List[str]
    type: str
    created_at: str

class ExpenseCreate(BaseModel):
    description: str = Field(..., description="Description of the expense")
    amount: float = Field(..., gt=0, description="Amount of the expense")
    paid_by: str = Field(..., description="ID of user who paid the expense")
    split_between: List[str] = Field(..., description="List of user IDs between whom the expense is split")
    category: Optional[str] = Field(None, description="Category of the expense")
    group_id: Optional[str] = Field(None, description="ID of the group this expense belongs to")

class ExpenseResponse(BaseModel):
    id: str
    description: str
    amount: float
    paid_by: str
    split_between: List[str]
    category: str
    group_id: Optional[str]
    created_at: str

class Debt(BaseModel):
    debtor: str
    creditor: str
    amount: float
    expense_id: Optional[str] = None
    status: str = "pending"

class DebtResponse(BaseModel):
    debtor: str
    creditor: str
    amount: float
    expense_id: Optional[str]
    status: str
    created_at: str

class Reminder(BaseModel):
    debtor: str
    creditor: str
    amount: float
    message: str
    status: str = "generated"

class ReminderResponse(BaseModel):
    id: str
    debtor: str
    creditor: str
    amount: float
    message: str
    status: str
    created_at: str

class SettlementResult(BaseModel):
    group_id: str
    balances: Dict[str, float]
    optimal_settlements: List[Debt]

class ReceiptScanResult(BaseModel):
    total_amount: Optional[float]
    date: str
    vendor: str
    raw_text: str
    success: bool
    message: str

# ===== SETTLEMENT OPTIMIZER =====
class SettlementOptimizer:
    @staticmethod
    def calculate_balances(expenses):
        """Calculate net balance for each user"""
        balances = {}
        
        for expense in expenses:
            payer = expense["paid_by"]
            amount = expense["amount"]
            split_count = len(expense["split_between"])
            share = amount / split_count if split_count > 0 else 0
            
            # Update payer's balance
            balances[payer] = balances.get(payer, 0) + amount
            
            # Update beneficiaries' balances
            for user_id in expense["split_between"]:
                balances[user_id] = balances.get(user_id, 0) - share
        
        return balances

    @staticmethod
    def minimize_transactions(balances):
        """Optimize settlements to minimize number of transactions"""
        debts = []
        
        # Convert balances to list of debts
        creditors = []
        debtors = []
        
        for user_id, balance in balances.items():
            if balance > 0:
                creditors.append((user_id, balance))
            elif balance < 0:
                debtors.append((user_id, -balance))
        
        # Sort creditors and debtors
        creditors.sort(key=lambda x: x[1], reverse=True)
        debtors.sort(key=lambda x: x[1], reverse=True)
        
        # Distribute debts
        i = j = 0
        while i < len(creditors) and j < len(debtors):
            creditor, cred_amt = creditors[i]
            debtor, deb_amt = debtors[j]
            
            settlement_amt = min(cred_amt, deb_amt)
            debts.append({
                "from": debtor,
                "to": creditor,
                "amount": round(settlement_amt, 2)
            })
            
            creditors[i] = (creditor, cred_amt - settlement_amt)
            debtors[j] = (debtor, deb_amt - settlement_amt)
            
            if creditors[i][1] < 0.01:  # Floating point tolerance
                i += 1
            if debtors[j][1] < 0.01:    # Floating point tolerance
                j += 1
        
        return debts

    @staticmethod
    def optimize_settlements(expenses):
        """Main method to calculate optimal settlements"""
        balances = SettlementOptimizer.calculate_balances(expenses)
        settlements = SettlementOptimizer.minimize_transactions(balances)
        
        return {
            "balances": balances,
            "optimal_settlements": settlements
        }

# ===== API ENDPOINTS =====
@app.get("/")
async def root():
    return {"message": "Expense Sharing System API"}

@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate):
    """Create a new user"""
    user_id = str(len(users_db) + 1)
    users_db[user_id] = {
        "id": user_id,
        **user.dict(),
        "created_at": datetime.now().isoformat()
    }
    return users_db[user_id]

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """Get user details"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[user_id]

@app.get("/users/", response_model=Dict[str, UserResponse])
async def list_users():
    """List all users"""
    return users_db

@app.post("/groups/", response_model=GroupResponse)
async def create_group(group: GroupCreate):
    """Create a new group"""
    # Validate that all member IDs exist
    for member_id in group.member_ids:
        if member_id not in users_db:
            raise HTTPException(status_code=400, detail=f"User {member_id} does not exist")
    
    group_id = str(len(groups_db) + 1)
    groups_db[group_id] = {
        "id": group_id,
        "name": group.name,
        "members": group.member_ids,
        "type": group.type,
        "created_at": datetime.now().isoformat()
    }
    return groups_db[group_id]

@app.get("/groups/{group_id}", response_model=GroupResponse)
async def get_group(group_id: str):
    """Get group details"""
    if group_id not in groups_db:
        raise HTTPException(status_code=404, detail="Group not found")
    return groups_db[group_id]

@app.get("/groups/", response_model=Dict[str, GroupResponse])
async def list_groups():
    """List all groups"""
    return groups_db

@app.patch("/groups/{group_id}", response_model=GroupResponse)
async def update_group_type(group_id: str, group_type: str):
    """Update the type of a group"""
    if group_id not in groups_db:
        raise HTTPException(status_code=404, detail="Group not found")
    
    groups_db[group_id]["type"] = group_type
    
    return groups_db[group_id]

@app.post("/expenses/", response_model=ExpenseResponse)
async def create_expense(expense: ExpenseCreate):
    """Create a new expense"""
    # Validate that paid_by user exists
    if expense.paid_by not in users_db:
        raise HTTPException(status_code=400, detail=f"User {expense.paid_by} does not exist")
    
    # Validate that all split_between users exist
    for user_id in expense.split_between:
        if user_id not in users_db:
            raise HTTPException(status_code=400, detail=f"User {user_id} does not exist")
    
    # Validate that group exists if provided
    if expense.group_id and expense.group_id not in groups_db:
        raise HTTPException(status_code=400, detail=f"Group {expense.group_id} does not exist")
    
    expense_id = str(len(expenses_db) + 1)
    
    # Use default category if not provided
    category = expense.category or "Miscellaneous"
    
    expenses_db[expense_id] = {
        "id": expense_id,
        **expense.dict(),
        "category": category,
        "created_at": datetime.now().isoformat()
    }
    
    # Calculate debts
    amount_per_person = expense.amount / len(expense.split_between)
    for debtor in expense.split_between:
        if debtor != expense.paid_by:
            debt_id = f"{expense_id}_{debtor}"
            debts_db[debt_id] = {
                "debtor": debtor,
                "creditor": expense.paid_by,
                "amount": amount_per_person,
                "expense_id": expense_id,
                "group_id": expense.group_id,
                "created_at": datetime.now().isoformat(),
                "status": "pending"
            }
    
    return expenses_db[expense_id]

@app.get("/expenses/{expense_id}", response_model=ExpenseResponse)
async def get_expense(expense_id: str):
    """Get expense details"""
    if expense_id not in expenses_db:
        raise HTTPException(status_code=404, detail="Expense not found")
    return expenses_db[expense_id]

@app.get("/expenses/", response_model=Dict[str, ExpenseResponse])
async def list_expenses(group_id: Optional[str] = None):
    """List all expenses, optionally filtered by group"""
    if group_id:
        return {eid: expense for eid, expense in expenses_db.items() if expense.get("group_id") == group_id}
    return expenses_db

@app.get("/groups/{group_id}/settlements", response_model=SettlementResult)
async def calculate_settlements(group_id: str):
    """Calculate optimal settlements for a group"""
    if group_id not in groups_db:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Get all expenses for this group
    group_expenses = [
        expense for expense in expenses_db.values() 
        if expense.get("group_id") == group_id
    ]
    
    # Calculate optimal settlements
    settlement_data = SettlementOptimizer.optimize_settlements(group_expenses)
    
    return {
        "group_id": group_id,
        **settlement_data
    }

@app.get("/groups/{group_id}/spending", response_model=dict)
async def get_group_spending(group_id: str):
    """Get spending breakdown by category for a specific group"""
    if group_id not in groups_db:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Get all expenses for this group
    group_expenses = [
        expense for expense in expenses_db.values() 
        if expense.get("group_id") == group_id
    ]
    
    # Calculate spending by category
    spending_by_category = {}
    for expense in group_expenses:
        category = expense.get("category", "Miscellaneous")
        amount = expense.get("amount", 0)
        spending_by_category[category] = spending_by_category.get(category, 0) + amount
    
    return {
        "group_id": group_id,
        "spending_by_category": spending_by_category,
        "total_spent": sum(spending_by_category.values())
    }

@app.get("/debts/", response_model=Dict[str, DebtResponse])
async def list_debts(user_id: Optional[str] = None, group_id: Optional[str] = None):
    """List all debts, optionally filtered by user or group"""
    filtered_debts = {}
    
    for debt_id, debt in debts_db.items():
        # Filter by user if specified
        if user_id and debt["debtor"] != user_id and debt["creditor"] != user_id:
            continue
        
        # Filter by group if specified
        if group_id and debt.get("group_id") != group_id:
            continue
            
        filtered_debts[debt_id] = debt
    
    return filtered_debts

@app.post("/debts/{debt_id}/settle", response_model=DebtResponse)
async def settle_debt(debt_id: str):
    """Mark a debt as settled"""
    if debt_id not in debts_db:
        raise HTTPException(status_code=404, detail="Debt not found")
    
    debts_db[debt_id]["status"] = "settled"
    debts_db[debt_id]["settled_at"] = datetime.now().isoformat()
    
    return debts_db[debt_id]

@app.get("/user/{user_id}/summary", response_model=dict)
async def get_user_summary(user_id: str):
    """Get a summary of a user's financial situation"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Calculate total owed to user
    total_owed = sum(
        debt["amount"] for debt in debts_db.values() 
        if debt["creditor"] == user_id and debt["status"] == "pending"
    )
    
    # Calculate total user owes
    total_owes = sum(
        debt["amount"] for debt in debts_db.values() 
        if debt["debtor"] == user_id and debt["status"] == "pending"
    )
    
    # Get recent expenses
    user_expenses = [
        expense for expense in expenses_db.values() 
        if expense["paid_by"] == user_id
    ]
    recent_expenses = sorted(user_expenses, key=lambda x: x["created_at"], reverse=True)[:5]
    
    return {
        "user_id": user_id,
        "total_owed_to_you": total_owed,
        "total_you_owe": total_owes,
        "net_balance": total_owed - total_owes,
        "recent_expenses": recent_expenses
    }

@app.post("/scan-receipt/", response_model=ReceiptScanResult)
async def scan_receipt_from_image():
    """Scan receipt from image.jpg file in the current directory"""
    try:
        # Path to the image.jpg file
        image_path = "image.jpg"
        
        # Check if the file exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="image.jpg file not found in the current directory")
        
        # Read the image file
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        # Scan the receipt
        receipt_data = ReceiptScanner.scan_receipt(image_data)
        
        return {
            "total_amount": receipt_data.get("total_amount"),
            "date": receipt_data.get("date", ""),
            "vendor": receipt_data.get("vendor", "Unknown Vendor"),
            "raw_text": receipt_data.get("raw_text", ""),
            "success": True,
            "message": "Receipt scanned successfully"
        }
        
    except Exception as e:
        logger.error(f"Error scanning receipt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scanning receipt: {str(e)}")

@app.post("/scan-receipt-create-expense/", response_model=dict)
async def scan_receipt_and_create_expense(
    paid_by: str,
    split_between: List[str],
    group_id: Optional[str] = None,
    category: Optional[str] = None
):
    """Scan receipt from image.jpg and automatically create an expense"""
    try:
        # First scan the receipt
        image_path = "image.jpg"
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="image.jpg file not found in the current directory")
        
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        
        receipt_data = ReceiptScanner.scan_receipt(image_data)
        
        if receipt_data.get("total_amount") is None:
            raise HTTPException(status_code=400, detail="Could not extract total amount from receipt")
        
        # Create expense from receipt data
        expense_data = ExpenseCreate(
            description=f"Receipt from {receipt_data.get('vendor', 'Unknown Vendor')}",
            amount=receipt_data["total_amount"],
            paid_by=paid_by,
            split_between=split_between,
            category=category or "Miscellaneous",
            group_id=group_id
        )
        
        # Validate that paid_by user exists
        if paid_by not in users_db:
            raise HTTPException(status_code=400, detail=f"User {paid_by} does not exist")
        
        # Validate that all split_between users exist
        for user_id in split_between:
            if user_id not in users_db:
                raise HTTPException(status_code=400, detail=f"User {user_id} does not exist")
        
        # Validate that group exists if provided
        if group_id and group_id not in groups_db:
            raise HTTPException(status_code=400, detail=f"Group {group_id} does not exist")
        
        expense_id = str(len(expenses_db) + 1)
        
        expenses_db[expense_id] = {
            "id": expense_id,
            **expense_data.dict(),
            "created_at": datetime.now().isoformat()
        }
        
        # Calculate debts
        amount_per_person = expense_data.amount / len(expense_data.split_between)
        for debtor in expense_data.split_between:
            if debtor != expense_data.paid_by:
                debt_id = f"{expense_id}_{debtor}"
                debts_db[debt_id] = {
                    "debtor": debtor,
                    "creditor": expense_data.paid_by,
                    "amount": amount_per_person,
                    "expense_id": expense_id,
                    "group_id": expense_data.group_id,
                    "created_at": datetime.now().isoformat(),
                    "status": "pending"
                }
        
        return {
            "success": True,
            "message": "Receipt scanned and expense created successfully",
            "receipt_data": receipt_data,
            "expense": expenses_db[expense_id]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scanning receipt and creating expense: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing receipt: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)