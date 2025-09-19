from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class UserCreate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None

class GroupCreate(BaseModel):
    name: str
    member_ids: List[str]

class ExpenseCreate(BaseModel):
    description: str
    amount: float
    paid_by: str
    split_between: List[str]
    category: Optional[str] = None

class Debt(BaseModel):
    debtor: str
    creditor: str
    amount: float
    expense_id: Optional[str] = None
    status: str = "pending"

class Reminder(BaseModel):
    debtor: str
    creditor: str
    amount: float
    message: str
    status: str = "generated"

class ReceiptScanResult(BaseModel):
    total_amount: Optional[float] = None
    date: str
    vendor: str
    raw_text: str
    category: str

class SettlementResult(BaseModel):
    group_id: str
    balances: Dict[str, float]
    optimal_settlements: List[Debt]