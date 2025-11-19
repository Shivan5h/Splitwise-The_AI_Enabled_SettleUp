from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import logging
import json
import os
import uvicorn

from receipt_scanner import ReceiptScanner
from expense_categorizer import ExpenseCategorizer  
from settlement_optimizer import SettlementOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Splitwise AI - Expense Sharing API",
    description="Scan → Categorize → Optimize Settlement",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
groups_db = {}
expenses_db = {}
users_db = {}

# Initialize AI categorizer (start with rule-based for faster startup)
expense_categorizer = ExpenseCategorizer(use_llm=False)
logger.info("Expense categorizer initialized (rule-based mode)")

# Data models
class User(BaseModel):
    id: str = Field(..., example="user_123")
    name: str = Field(..., example="John Doe")
    email: str = Field(..., example="john@email.com")

class GroupCreate(BaseModel):
    name: str = Field(..., example="Trip to Goa")
    members: List[User]

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None

class ExpenseCreate(BaseModel):
    description: str = Field(..., example="Dinner at restaurant")
    amount: float = Field(..., gt=0, example=150.50)
    paid_by_user_id: str
    split_among_user_ids: List[str]
    group_id: str
    category: Optional[str] = None

# API Endpoints

@app.get("/", response_model=ApiResponse)
async def health_check():
    """Health check - verify API is running"""
    return ApiResponse(
        success=True,
        message="Splitwise AI API is running",
        data={
            "version": "2.0.0",
            "workflow": "Scan → Categorize → Optimize",
            "endpoints": 8,
            "ai_enabled": expense_categorizer.use_llm
        }
    )

@app.post("/groups/create", response_model=ApiResponse)
async def create_group(group: GroupCreate):
    """Create new expense-sharing group"""
    try:
        group_id = f"group_{len(groups_db) + 1}"
        
        # Store members in users database
        for member in group.members:
            if member.id not in users_db:
                users_db[member.id] = member.dict()
        
        # Create group record
        group_record = {
            "id": group_id,
            "name": group.name,
            "members": [member.dict() for member in group.members],
            "created_at": datetime.now().isoformat(),
            "total_expenses": 0,
            "total_amount": 0.0
        }
        
        groups_db[group_id] = group_record
        logger.info(f"Group created: {group_id}")
        
        return ApiResponse(
            success=True,
            message=f"Group '{group.name}' created successfully",
            data={"group": group_record}
        )
        
    except Exception as e:
        logger.error(f"Group creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/groups/{group_id}", response_model=ApiResponse)
async def get_group(group_id: str):
    """Get group details and members"""
    if group_id not in groups_db:
        raise HTTPException(status_code=404, detail="Group not found")
    
    return ApiResponse(
        success=True,
        message="Group found",
        data={"group": groups_db[group_id]}
    )

@app.post("/scan-receipt", response_model=ApiResponse)
async def scan_receipt_and_create_expense(
    file: UploadFile = File(...),
    group_id: str = Form(...),
    paid_by_user_id: str = Form(...),
    split_among_user_ids: str = Form(...)
):
    """Main feature: Scan receipt -> Auto-categorize -> Create expense"""
    try:
        # Validate inputs
        if group_id not in groups_db:
            raise HTTPException(status_code=404, detail="Group not found")
        
        split_users = json.loads(split_among_user_ids)
        
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Please upload a valid image")
        
        # Step 1: OCR scan receipt
        logger.info("Scanning receipt...")
        image_data = await file.read()
        receipt_data = ReceiptScanner.scan_receipt(image_data)
        
        if not receipt_data.get("total_amount") or receipt_data.get("total_amount") == 0:
            return ApiResponse(
                success=False,
                message="Could not extract amount from receipt. Try a clearer image or add manually.",
                data={
                    "raw_text": receipt_data.get("raw_text", ""),
                    "vendor": receipt_data.get("vendor", "")
                }
            )
        
        # Step 2: AI categorization
        logger.info("AI categorizing...")
        vendor = receipt_data.get("vendor", "Unknown Vendor")
        description = f"Receipt from {vendor}"
        
        try:
            category = expense_categorizer.categorize(description=description, vendor=vendor)
        except Exception as e:
            logger.warning(f"Categorization failed: {e}")
            category = "Gifts & Miscellaneous"
        
        # Step 3: Create expense
        expense_id = f"exp_{len(expenses_db) + 1}"
        amount = receipt_data["total_amount"]
        
        expense = {
            "id": expense_id,
            "description": description,
            "amount": amount,
            "paid_by_user_id": paid_by_user_id,
            "split_among_user_ids": split_users,
            "group_id": group_id,
            "category": category,
            "receipt_data": receipt_data,
            "created_at": datetime.now().isoformat()
        }
        
        expenses_db[expense_id] = expense
        
        # Update group stats
        groups_db[group_id]["total_expenses"] += 1
        groups_db[group_id]["total_amount"] += amount
        
        logger.info(f"Expense created: ₹{amount} -> {category}")
        
        return ApiResponse(
            success=True,
            message=f"Receipt scanned! ₹{amount} auto-categorized as '{category}'",
            data={
                "expense": expense,
                "amount": amount,
                "vendor": vendor,
                "category": category
            }
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid user IDs format")
    except Exception as e:
        logger.error(f"Receipt processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/expenses/manual", response_model=ApiResponse)
async def create_manual_expense(expense: ExpenseCreate):
    """Create manual expense (backup method)"""
    try:
        if expense.group_id not in groups_db:
            raise HTTPException(status_code=404, detail="Group not found")
        
        # Auto-categorize
        category = expense.category or expense_categorizer.categorize(expense.description)
        
        expense_id = f"exp_{len(expenses_db) + 1}"
        expense_dict = {
            "id": expense_id,
            **expense.dict(),
            "category": category,
            "created_at": datetime.now().isoformat()
        }
        
        expenses_db[expense_id] = expense_dict
        
        # Update group stats
        groups_db[expense.group_id]["total_expenses"] += 1
        groups_db[expense.group_id]["total_amount"] += expense.amount
        
        return ApiResponse(
            success=True,
            message=f"Manual expense ₹{expense.amount} added",
            data={"expense": expense_dict, "category": category}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/groups/{group_id}/expenses", response_model=ApiResponse)
async def get_group_expenses(group_id: str):
    """Get all expenses for a group"""
    if group_id not in groups_db:
        raise HTTPException(status_code=404, detail="Group not found")
    
    # Get expenses for this group
    group_expenses = [
        expense for expense in expenses_db.values()
        if expense["group_id"] == group_id
    ]
    
    # Calculate category breakdown
    category_totals = {}
    for expense in group_expenses:
        category = expense["category"]
        category_totals[category] = category_totals.get(category, 0) + expense["amount"]
    
    # Sort by date (newest first)
    group_expenses.sort(key=lambda x: x["created_at"], reverse=True)
    
    return ApiResponse(
        success=True,
        message=f"Found {len(group_expenses)} expenses",
        data={
            "expenses": group_expenses,
            "category_breakdown": category_totals,
            "total_amount": sum(exp["amount"] for exp in group_expenses)
        }
    )

@app.post("/groups/{group_id}/calculate-settlement", response_model=ApiResponse)
async def calculate_settlement(group_id: str):
    """Calculate optimal settlement to minimize transactions"""
    try:
        if group_id not in groups_db:
            raise HTTPException(status_code=404, detail="Group not found")
        
        # Get group expenses
        group_expenses = [
            expense for expense in expenses_db.values()
            if expense["group_id"] == group_id
        ]
        
        if not group_expenses:
            return ApiResponse(
                success=True,
                message="No expenses to settle",
                data={"settlements": [], "balances": {}}
            )
        
        # Prepare data for settlement optimizer
        adapted_expenses = []
        for expense in group_expenses:
            adapted_expenses.append({
                "paid_by": expense["paid_by_user_id"],
                "amount": expense["amount"],
                "split_between": expense["split_among_user_ids"]
            })
        
        # Calculate optimal settlements
        logger.info("Calculating optimal settlements...")
        settlement_result = SettlementOptimizer.optimize_settlements(adapted_expenses)
        
        # Format for UI
        group_members = {m["id"]: m for m in groups_db[group_id]["members"]}
        settlements = []
        
        for settlement in settlement_result["optimal_settlements"]:
            from_user = group_members.get(settlement["from"], {"name": f"User {settlement['from']}"})
            to_user = group_members.get(settlement["to"], {"name": f"User {settlement['to']}"})
            
            settlements.append({
                "from_user_id": settlement["from"],
                "to_user_id": settlement["to"],
                "from_user": from_user,
                "to_user": to_user,
                "amount": settlement["amount"],
                "message": f"{from_user['name']} pays ₹{settlement['amount']:.2f} to {to_user['name']}"
            })
        
        logger.info(f"Settlement optimized: {len(settlements)} transactions")
        
        return ApiResponse(
            success=True,
            message=f"Settlement calculated: {len(settlements)} payments needed",
            data={
                "settlements": settlements,
                "balances": settlement_result["balances"],
                "total_transactions": len(settlements)
            }
        )
        
    except Exception as e:
        logger.error(f"Settlement calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories", response_model=ApiResponse)
async def get_categories():
    """Get available expense categories"""
    return ApiResponse(
        success=True,
        message="Categories retrieved",
        data={
            "categories": ExpenseCategorizer.CATEGORIES,
            "examples": ExpenseCategorizer.CATEGORY_EXAMPLES,
            "ai_powered": expense_categorizer.use_llm
        }
    )

# Startup
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 SPLITWISE AI - EXPENSE SHARING API")
    print("="*60)
    print("📱 Endpoints: 8")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🌐 Server: http://0.0.0.0:8000")
    print("🎯 Workflow: Scan → Categorize → Optimize")
    print("📱 Kotlin App: Use http://10.0.2.2:8000 (Android Emulator)")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)