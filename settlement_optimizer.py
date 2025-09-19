from typing import Dict, List, Any

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