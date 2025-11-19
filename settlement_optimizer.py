from typing import Dict, List, Any

class SettlementOptimizer:
    @staticmethod
    def calculate_balances(expenses):
        """Calculate net balance for each user"""
        balances = {}
        
        for expense in expenses:
            payer = expense.get("paid_by")
            amount = expense.get("amount", 0)
            split_between = expense.get("split_between", [])
            
            if not payer or not split_between or amount <= 0:
                continue
            
            split_count = len(split_between)
            share = round(amount / split_count, 2) if split_count > 0 else 0
            
            # Payer receives money back (paid more than their share)
            balances[payer] = balances.get(payer, 0.0) + amount - share
            
            # Others owe their share
            for user_id in split_between:
                if user_id != payer:
                    balances[user_id] = balances.get(user_id, 0.0) - share
        
        return balances

    @staticmethod
    def minimize_transactions(balances):
        """Optimize settlements using greedy algorithm to minimize transactions"""
        # Filter out near-zero balances
        creditors = [(uid, round(bal, 2)) for uid, bal in balances.items() if bal > 0.01]
        debtors = [(uid, round(-bal, 2)) for uid, bal in balances.items() if bal < -0.01]
        
        if not creditors or not debtors:
            return []
        
        # Sort by amount (largest first) for optimal matching
        creditors.sort(key=lambda x: x[1], reverse=True)
        debtors.sort(key=lambda x: x[1], reverse=True)
        
        settlements = []
        i = j = 0
        
        while i < len(creditors) and j < len(debtors):
            creditor_id, cred_amt = creditors[i]
            debtor_id, deb_amt = debtors[j]
            
            # Settle the minimum of what's owed/owed to
            settle_amt = min(cred_amt, deb_amt)
            
            if settle_amt > 0.01:  # Only add if meaningful amount
                settlements.append({
                    "from": debtor_id,
                    "to": creditor_id,
                    "amount": round(settle_amt, 2)
                })
            
            # Update remaining balances
            creditors[i] = (creditor_id, round(cred_amt - settle_amt, 2))
            debtors[j] = (debtor_id, round(deb_amt - settle_amt, 2))
            
            # Move to next if settled
            if creditors[i][1] < 0.01:
                i += 1
            if debtors[j][1] < 0.01:
                j += 1
        
        return settlements

    @staticmethod
    def optimize_settlements(expenses):
        """Main method to calculate optimal settlements"""
        balances = SettlementOptimizer.calculate_balances(expenses)
        settlements = SettlementOptimizer.minimize_transactions(balances)
        
        return {
            "balances": balances,
            "optimal_settlements": settlements
        }