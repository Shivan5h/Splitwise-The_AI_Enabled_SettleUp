import re 
import numpy as np
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ML dependencies only when needed
HAS_ML_DEPS = True
SentenceTransformer = None
cosine_similarity = None

def _load_ml_dependencies():
    global SentenceTransformer, cosine_similarity, HAS_ML_DEPS
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as ST
            from sklearn.metrics.pairwise import cosine_similarity as cs
            SentenceTransformer = ST
            cosine_similarity = cs
            logger.info("ML dependencies loaded successfully")
        except ImportError as e:
            HAS_ML_DEPS = False
            logger.warning(f"ML dependencies not available: {e}. ML-based categorization will be disabled.")
            return False
    return HAS_ML_DEPS

class ExpenseCategorizer:
    CATEGORIES = [
        "Housing & Shared Living",
        "Food & Groceries", 
        "Travel & Trips",
        "Events & Entertainment",
        "Household & Essentials",
        "Work / Education",
        "Health & Fitness",
        "Gifts & Miscellaneous"
    ]
    
    CATEGORY_EXAMPLES = {
        "Housing & Shared Living": [
            "rent payment for apartment",
            "electricity bill for shared house",
            "internet subscription for flat",
            "water utility bill",
            "furniture for common area"
        ],
        "Food & Groceries": [
            "groceries from supermarket",
            "dinner at restaurant with friends",
            "coffee at cafe with colleagues",
            "snacks for movie night",
            "alcohol for party"
        ],
        "Travel & Trips": [
            "uber ride to airport",
            "flight tickets for vacation",
            "hotel booking for trip",
            "train tickets for group outing",
            "gas for road trip"
        ],
        "Events & Entertainment": [
            "movie tickets for cinema",
            "netflix subscription shared",
            "concert tickets with friends",
            "bowling night with group",
            "escape room booking"
        ],
        "Household & Essentials": [
            "cleaning supplies for house",
            "toiletries from drugstore",
            "shared appliances purchase",
            "kitware for common kitchen",
            "household essentials"
        ],
        "Work / Education": [
            "office supplies for team",
            "online course subscription",
            "books for study group",
            "software license shared",
            "stationery for project"
        ],
        "Health & Fitness": [
            "gym membership shared",
            "sports equipment for group",
            "badminton court rental",
            "yoga class with friends",
            "first aid kit for trips"
        ],
        "Gifts & Miscellaneous": [
            "birthday gift for friend",
            "group donation to charity",
            "miscellaneous shared expenses",
            "unexpected group costs",
            "general shared purchases"
        ]
    }
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.model = None
        self.category_embeddings = {}
        
        # Don't load ML dependencies at init, load them when needed
        if use_llm:
            logger.info("ML-based categorization requested. Will load dependencies when needed.")

    def _initialize_model(self):
        if not _load_ml_dependencies():
            self.use_llm = False
            return False
            
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model")

            self.category_embeddings = {}
            for category, examples in self.CATEGORY_EXAMPLES.items():
                example_texts = [f"{category}: {example}" for example in examples]
                embeddings = self.model.encode(example_texts)
                self.category_embeddings[category] = embeddings
            return True
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.use_llm = False
            return False

    def categorize_with_llm(self, description: str) -> str:
        if not self.use_llm or not description:
            return self.categorize_rule_based(description)
        
        # Initialize model on first use
        if self.model is None:
            if not self._initialize_model():
                return self.categorize_rule_based(description)
        
        try:
            query_embedding = self.model.encode([description])
            
            best_category = None
            best_similarity = -1
            
            for category, embeddings in self.category_embeddings.items():
                similarities = cosine_similarity(query_embedding, embeddings)
                max_similarity = np.max(similarities)
                
                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    best_category = category

            if best_similarity > 0.3: 
                return best_category
            else:
                return self.categorize_rule_based(description)
                
        except Exception as e:
            logger.error(f"LLM categorization failed: {e}")
            return self.categorize_rule_based(description)
    
    def categorize_rule_based(self, description: str, vendor: str = None) -> str:
        if not description:
            return "Gifts & Miscellaneous"
        
        text = (description + " " + (vendor or "")).lower()
        
        # Enhanced keyword mapping with common vendor names
        keyword_mapping = {
            "Housing & Shared Living": ["rent", "lease", "electric", "electricity", "water", "gas", "internet", "wifi", "utility", "bill"],
            "Food & Groceries": ["grocery", "restaurant", "cafe", "food", "meal", "dinner", "lunch", "breakfast", "coffee", "mcdonald", "kfc", "pizza", "burger", "supermarket", "walmart", "store"],
            "Travel & Trips": ["uber", "lyft", "ola", "taxi", "cab", "flight", "airline", "hotel", "airbnb", "train", "bus", "travel", "trip", "petrol", "fuel"],
            "Events & Entertainment": ["movie", "cinema", "netflix", "spotify", "concert", "game", "bowling", "event", "ticket", "theatre", "show"],
            "Household & Essentials": ["cleaning", "toilet", "detergent", "supply", "furniture", "appliance", "ikea", "home", "kitchen"],
            "Work / Education": ["office", "study", "course", "book", "amazon", "software", "project", "stationery", "school", "college"],
            "Health & Fitness": ["gym", "sports", "fitness", "medical", "health", "doctor", "medicine", "pharmacy", "yoga"],
            "Gifts & Miscellaneous": ["gift", "donation", "charity", "present"]
        }
        
        # Check vendor names and description
        for category, keywords in keyword_mapping.items():
            if any(keyword in text for keyword in keywords):
                return category
        
        return "Gifts & Miscellaneous"
    
    def categorize(self, description: str, vendor: str = None, use_llm: bool = None) -> str:
        use_llm = use_llm if use_llm is not None else self.use_llm
        
        if use_llm:
            return self.categorize_with_llm(description)
        else:
            return self.categorize_rule_based(description, vendor)
    
    def batch_categorize(self, descriptions: List[str], use_llm: bool = None) -> List[str]:
        use_llm = use_llm if use_llm is not None else self.use_llm
        
        if use_llm and self.use_llm:
            # Initialize model on first use
            if self.model is None:
                if not self._initialize_model():
                    return [self.categorize_rule_based(desc) for desc in descriptions]
            
            try:
                embeddings = self.model.encode(descriptions)
                categories = []
                
                for i, embedding in enumerate(embeddings):
                    best_category = None
                    best_similarity = -1
                    
                    for category, cat_embeddings in self.category_embeddings.items():
                        similarities = cosine_similarity([embedding], cat_embeddings)
                        max_similarity = np.max(similarities)
                        
                        if max_similarity > best_similarity:
                            best_similarity = max_similarity
                            best_category = category
                    
                    if best_similarity > 0.3:
                        categories.append(best_category)
                    else:
                        categories.append(self.categorize_rule_based(descriptions[i]))
                
                return categories
            except Exception as e:
                logger.error(f"Batch categorization failed: {e}")
        
        return [self.categorize_rule_based(desc) for desc in descriptions]

categorizer = ExpenseCategorizer()
