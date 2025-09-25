import streamlit as st
import requests
import json
from PIL import Image
import os
from datetime import datetime
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="Splitwise - Receipt Scanner",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_connection():
    """Check if the FastAPI server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        return response.status_code == 200
    except:
        return False

def get_users():
    """Fetch all users from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/users/")
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

def get_groups():
    """Fetch all groups from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/groups/")
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}

def scan_receipt():
    """Call the receipt scanning API"""
    try:
        response = requests.post(f"{API_BASE_URL}/scan-receipt/")
        return response.json(), response.status_code == 200
    except Exception as e:
        return {"error": str(e)}, False

def scan_and_create_expense(paid_by, split_between, group_id=None, category=None):
    """Call the scan receipt and create expense API"""
    try:
        payload = {
            "paid_by": paid_by,
            "split_between": split_between,
            "group_id": group_id,
            "category": category
        }
        response = requests.post(f"{API_BASE_URL}/scan-receipt-create-expense/", json=payload)
        return response.json(), response.status_code == 200
    except Exception as e:
        return {"error": str(e)}, False

# Main App
def main():
    st.markdown('<h1 class="main-header">🧾 Splitwise Receipt Scanner</h1>', unsafe_allow_html=True)
    
    # Check API connection
    if not check_api_connection():
        st.markdown(
            '<div class="error-box">❌ <strong>API Connection Error:</strong> '
            'FastAPI server is not running. Please start the server with: <code>python main.py</code></div>',
            unsafe_allow_html=True
        )
        st.stop()
    
    st.markdown(
        '<div class="success-box">✅ <strong>API Connected:</strong> FastAPI server is running successfully!</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar for navigation
    st.sidebar.title("🎛️ Control Panel")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["Receipt Scanner", "Create Expense from Receipt", "Manage Users & Groups"]
    )
    
    if app_mode == "Receipt Scanner":
        receipt_scanner_page()
    elif app_mode == "Create Expense from Receipt":
        create_expense_page()
    elif app_mode == "Manage Users & Groups":
        manage_users_groups_page()

def receipt_scanner_page():
    st.header("📱 Receipt Scanner")
    
    # Check if image.jpg exists
    if os.path.exists("image.jpg"):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Current Receipt Image")
            try:
                image = Image.open("image.jpg")
                st.image(image, caption="image.jpg", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
        
        with col2:
            st.subheader("🔍 Scan Results")
            
            if st.button("🚀 Scan Receipt", type="primary", use_container_width=True):
                with st.spinner("Scanning receipt... Please wait"):
                    result, success = scan_receipt()
                
                if success:
                    st.markdown('<div class="success-box">✅ <strong>Receipt scanned successfully!</strong></div>', unsafe_allow_html=True)
                    
                    # Display results in a nice format
                    st.subheader("📊 Extracted Information")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("💰 Total Amount", f"₹{result.get('total_amount', 'N/A')}")
                        st.metric("📅 Date", result.get('date', 'N/A'))
                    
                    with col_b:
                        st.metric("🏪 Vendor", result.get('vendor', 'Unknown'))
                        st.metric("✅ Status", "Success" if result.get('success') else "Failed")
                    
                    # Raw text in expandable section
                    with st.expander("📝 Raw Extracted Text"):
                        st.text_area("", value=result.get('raw_text', ''), height=200, disabled=True)
                    
                    # Store result in session state for use in expense creation
                    st.session_state.last_scan_result = result
                    
                else:
                    st.markdown(f'<div class="error-box">❌ <strong>Scan failed:</strong> {result.get("error", "Unknown error")}</div>', unsafe_allow_html=True)
    
    else:
        st.markdown(
            '<div class="info-box">📷 <strong>No Receipt Image Found:</strong> '
            'Please place an image file named <code>image.jpg</code> in the project directory.</div>',
            unsafe_allow_html=True
        )
        
        # File uploader as alternative
        st.subheader("📤 Upload Receipt Image")
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Save uploaded file as image.jpg
            with open("image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("✅ Image uploaded successfully! Please refresh the page.")
            st.experimental_rerun()

def create_expense_page():
    st.header("💸 Create Expense from Receipt")
    
    # Check if we have a recent scan result
    if 'last_scan_result' in st.session_state:
        result = st.session_state.last_scan_result
        st.markdown(
            f'<div class="info-box">💡 <strong>Using last scan result:</strong> '
            f'₹{result.get("total_amount", "N/A")} from {result.get("vendor", "Unknown")}</div>',
            unsafe_allow_html=True
        )
    
    # Get users and groups
    users = get_users()
    groups = get_groups()
    
    if not users:
        st.markdown(
            '<div class="error-box">❌ <strong>No Users Found:</strong> '
            'Please create users first using the FastAPI endpoints or the Manage Users & Groups section.</div>',
            unsafe_allow_html=True
        )
        return
    
    # Create expense form
    st.subheader("📝 Expense Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User selection
        user_options = {f"{user['name']} ({user['email']})": user_id for user_id, user in users.items()}
        paid_by_display = st.selectbox("💳 Who paid?", options=list(user_options.keys()))
        paid_by = user_options[paid_by_display]
        
        # Split between selection
        split_between_display = st.multiselect(
            "👥 Split between (select multiple)",
            options=list(user_options.keys()),
            default=[paid_by_display]
        )
        split_between = [user_options[display] for display in split_between_display]
    
    with col2:
        # Group selection
        group_options = {"No Group": None}
        if groups:
            group_options.update({f"{group['name']} ({group['type']})": group_id for group_id, group in groups.items()})
        
        group_display = st.selectbox("👥 Group (optional)", options=list(group_options.keys()))
        group_id = group_options[group_display]
        
        # Category selection
        categories = ["Food & Dining", "Transportation", "Shopping", "Entertainment", "Bills & Utilities", "Miscellaneous"]
        category = st.selectbox("🏷️ Category", options=categories)
    
    # Create expense button
    if st.button("🚀 Scan Receipt & Create Expense", type="primary", use_container_width=True):
        if not split_between:
            st.error("❌ Please select at least one person to split the expense between.")
            return
        
        with st.spinner("Scanning receipt and creating expense... Please wait"):
            result, success = scan_and_create_expense(
                paid_by=paid_by,
                split_between=split_between,
                group_id=group_id,
                category=category
            )
        
        if success:
            st.markdown('<div class="success-box">✅ <strong>Expense created successfully!</strong></div>', unsafe_allow_html=True)
            
            # Display expense details
            expense = result.get('expense', {})
            receipt_data = result.get('receipt_data', {})
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("💰 Amount", f"₹{expense.get('amount', 'N/A')}")
            with col_b:
                st.metric("👥 Split Between", f"{len(split_between)} people")
            with col_c:
                st.metric("💳 Amount per Person", f"₹{expense.get('amount', 0) / len(split_between):.2f}")
            
            # Expense details
            st.subheader("📋 Expense Summary")
            expense_df = pd.DataFrame([{
                "ID": expense.get('id'),
                "Description": expense.get('description'),
                "Amount": f"₹{expense.get('amount')}",
                "Paid By": users.get(expense.get('paid_by'), {}).get('name', 'Unknown'),
                "Category": expense.get('category'),
                "Date": expense.get('created_at', '').split('T')[0] if expense.get('created_at') else 'N/A'
            }])
            st.dataframe(expense_df, use_container_width=True)
            
        else:
            st.markdown(f'<div class="error-box">❌ <strong>Failed to create expense:</strong> {result.get("detail", "Unknown error")}</div>', unsafe_allow_html=True)

def manage_users_groups_page():
    st.header("👥 Manage Users & Groups")
    
    tab1, tab2 = st.tabs(["👤 Users", "👥 Groups"])
    
    with tab1:
        st.subheader("Current Users")
        users = get_users()
        
        if users:
            users_data = []
            for user_id, user in users.items():
                users_data.append({
                    "ID": user_id,
                    "Name": user.get('name'),
                    "Email": user.get('email'),
                    "Phone": user.get('phone', 'N/A'),
                    "Created": user.get('created_at', '').split('T')[0] if user.get('created_at') else 'N/A'
                })
            
            df = pd.DataFrame(users_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No users found. Create users using the FastAPI endpoints.")
    
    with tab2:
        st.subheader("Current Groups")
        groups = get_groups()
        
        if groups:
            groups_data = []
            for group_id, group in groups.items():
                users = get_users()
                member_names = [users.get(member_id, {}).get('name', f'User {member_id}') for member_id in group.get('members', [])]
                
                groups_data.append({
                    "ID": group_id,
                    "Name": group.get('name'),
                    "Type": group.get('type'),
                    "Members": ', '.join(member_names),
                    "Member Count": len(group.get('members', [])),
                    "Created": group.get('created_at', '').split('T')[0] if group.get('created_at') else 'N/A'
                })
            
            df = pd.DataFrame(groups_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No groups found. Create groups using the FastAPI endpoints.")

if __name__ == "__main__":
    main()