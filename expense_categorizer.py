import re 
import numpy as np
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    Hash_ML_DELPS = True
except ImportError:
    Hash_ML_DELPS = False
    logger.warning("sentence_transformers or sklearn not installed. ML-based categorization will be disabled.")

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
        self.use_llm = use_llm and Hash_ML_DELPS
        if self.use_llm:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.category_embeddings = self.model.encode(
                [example for examples in self.CATEGORY_EXAMPLES.values() for example in examples]
            )
            self.category_map = []
            for category, examples in self.CATEGORY_EXAMPLES.items():
                self.category_map.extend([category] * len(examples))
        else:
            if use_llm and not Hash_ML_DELPS:
                logger.warning("ML-based categorization is disabled due to missing dependencies.")
            self.model = None

        def _initialize_model(self):
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer model")

                self.category_embeddings = {}
                for category, examples in self.CATEGORY_EXAMPLES.items():
                    example_texts = [f"{category}: {example}" for example in examples]
                    embeddings = self.model.encode(example_texts)
                    self.category_embeddings[category] = embeddings
                    
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                self.use_llm = False

        def categorize_with_llm(self, description: str) -> str:
            if not self.use_llm or not description:
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
                
            description_lower = description.lower()
            
            # Keyword mapping (simplified version)
            keyword_mapping = {
                "Housing & Shared Living": ["rent", "lease", "electric", "water", "gas", "internet", "wifi", "utility"],
                "Food & Groceries": ["grocery", "restaurant", "cafe", "food", "meal", "dinner", "lunch", "coffee"],
                "Travel & Trips": ["uber", "lyft", "taxi", "flight", "hotel", "train", "bus", "travel"],
                "Events & Entertainment": ["movie", "netflix", "concert", "game", "bowling", "event", "ticket"],
                "Household & Essentials": ["cleaning", "toilet", "supply", "furniture", "appliance"],
                "Work / Education": ["office", "study", "course", "book", "software", "project"],
                "Health & Fitness": ["gym", "sports", "fitness", "medical", "health"],
                "Gifts & Miscellaneous": ["gift", "donation", "charity"]
            }
            
            for category, keywords in keyword_mapping.items():
                if any(keyword in description_lower for keyword in keywords):
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

categorizer = GroupExpenseCategorizerLLM()
